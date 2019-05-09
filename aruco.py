
import os
import time
import logging

import cv2
import imutils
import numpy as np
from cv2 import aruco
from pyrecord import Record
import threading
from rpyc.utils.server import ThreadedServer

import config
import rpcSend
import rpcReceive
import calibration
import watchDog


######################################################
# this task can run as a slave (controlled by rpc commands) or standAlone
######################################################
standAloneMode = False

point = Record.create_type('point', 'x', 'y')

markerType = Record.create_type('markerType', 'type', 'length', 'separation', 'orthoStopDist', 'allowedIds')
# adapt length to get a correct distance
# include distance of cart front to center (330) for the orthoStopDist
markerTypes = [markerType('dockingMarker', 100, 0.5, 600, [10]),
               markerType('dockingDetail', 60, 0.5, 0, [11]),
               markerType('objectMarker', 100, 1, 600, [0,1,2])]


# ARUCO_PORT = 20001
DOCKING_MARKER_ID = 10
DOCKING_DETAIL_ID = 11

DISTANCE_CART_CENTER_CAM = 330  # mm
MARKER_XOFFSET_CORRECTION = -22

CARTCAM_X_ANGLE = 70.0
CARTCAM_X_RESOLUTION = 640

CARTCAM_Y_ANGLE = 50
CARTCAM_Y_RESOLUTION = 480

EYECAM_X_ANGLE = 70.0
EYECAM_X_RESOLUTION = 640

EYECAM_Y_ANGLE =43
EYECAM_Y_RESOLUTION = 480

# calibrateCamera
# createMarkers()
# calibration.createCharucoBoard()
# calibration.takeCalibrationPictures()
# calibration.calibrateCamera()

# calibration.calibrateCameraCharuco()
# exit()


arucoParams = aruco.DetectorParameters_create()
# this parameter creates a border around each each cell in the cell grid
arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.25     # default = 0.13
arucoParams.minDistanceToBorder = 2

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

navManager = None
cap = None
lastImg = None


def log(msg):
    if navManager is not None:
        navManager.root.recordLog("aruco - " + msg)
    print(msg)


def lookForMarkers(camera, markerIds, camYaw):
    foundMarkers = []

    if camera == "EYE_CAM":
        img = config.eyecamImg

    elif camera == "CART_CAM":
        img = config.cartcamImg

    else:
        config.log(f"requested camera {camera} not handled in call")
        return []

    # check for marker
    foundIds, corners = findMarkers(img, show=False)

    if len(markerIds) > 0 and foundIds is None:
        config.log(f"none of the requested markers found in image")
        return []

    if foundIds is not None:
        #config.log(f"markers found: {foundIds}")
        for markerIndex, foundId in enumerate(foundIds):
            if len(markerIds) == 0 or foundId in markerIds:
                try:
                    markerInfo = calculateMarkerFacts(corners[markerIndex], foundId, camera)
                except Exception as e:
                    config.log(f"error in calculateMarkerFacts: {e}")
                    markerInfo = None
                if markerInfo is not None:
                    config.log(f"markerId: {markerInfo['markerId']}, distance: {markerInfo['distanceCamToMarker']}, angleToMarker: {markerInfo['angleToMarker']}, markerDegrees: {markerInfo['markerDegrees']}")
                    foundMarkers.append(markerInfo)

    return foundMarkers


def takeEyecamImage():

    if config.eyecam is None:
        config.log(f"try to connect with eye cam", publish=False)
        config.eyecam = cv2.VideoCapture(0)
    if config.eyecam is None:
        config.log(f"ERROR: could not connect with VideoCapture(0)")
        return None

    _, _ = config.eyecam.read()
    ret, config.eyecamImg = config.eyecam.read()
    if ret:
        config.log("eyecam image successfully taken")
        config.eyecamImg = imutils.rotate_bound(config.eyecamImg, 90)
        return config.eyecamImg

    else:
        config.log("error taking eyecam image")
        return None


def takeCartcamImage():

    if config.cartcam is None:
        config.log(f"try to connect with cart cam", publish=False)
        config.cartcam = cv2.VideoCapture(1)
    if config.cartcam is None:
        config.log(f"could not connect with VideoCapture(1)")

    # first image may be previously seen view
    for _ in range(5):
        _, _ = config.cartcam.read()
    ret, config.cartcamImg = config.cartcam.read()
    if ret:
        config.cartcamImg = imutils.rotate_bound(config.cartcamImg, 180)
        config.log("cartcam image successfully taken")
        return config.cartcamImg
    else:
        config.log("error taking cartcam image")
        return None


def arucoTerminate():
    log("stopping arucoServer")
    print("termination request received")
    time.sleep(2)
    raise SystemExit()


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.atan2(-R[1, 2], R[1, 1])
        y = np.atan2(-R[2, 0], sy)
        z = 0

    return np.array([y, x, z])


# calculate a cartYaw for cart to be positioned <distance> in front of the marker
def evalDegreesDistToCartTarget(degreesToMarker, distance, markerDegrees):
    log(
        f"evalDegreesDistToCartDestination degreesToMarker: {degreesToMarker:.0f}, distance: {distance:.0f}, markerDegrees: {markerDegrees:.0f}")

    # Position x,y of camera
    p1 = point(0, 0)

    # Center of Marker
    p2 = point(0, 0)
    p2.x = int(distance * np.cos(np.radians(degreesToMarker)))
    p2.y = int(distance * np.sin(np.radians(degreesToMarker)))
    # print(p2)

    # angle between marker and cart destination point (includes markerDegrees)
    beta = degreesToMarker - 90 + markerDegrees

    # cart destination point orthogonal in front of marker with offset
    p3 = point(0, 0)
    p3.x = int(p2.x + (distance * np.sin(np.radians(beta))))
    p3.y = int(p2.y - (distance * np.cos(np.radians(beta))))
    # print(p3)

    # angle to cart point in relation to degreesToMarker
    degreesToCartTarget = 180 + np.degrees(np.arctan(p3.y / p3.x))
    distToCartTarget = np.sqrt(np.power(p3.x, 2) + np.power(p3.y, 2))

    log(
        f"degreesToCartTarget: {degreesToCartTarget:.0f}, distToCartTarget {distToCartTarget:.0f}, markerPos: {p2}, cartTargetAngle: {beta:.0f}, cartTargetPos: {p3}")

    return degreesToCartTarget, distToCartTarget


def getAvgHue(img):
    # convert to hsv for color comparison
    imgH = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avgHsvByCol = np.average(imgH, axis=0)
    avgHsv = np.average(avgHsvByCol, axis=0)
    return avgHsv[0]  # hue


# search for marker in frame
def findMarkers(img, show):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # aruco.detectMarkers() requires gray image

    # timestr = datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")
    # cv2.imwrite("images/" + timestr + ".jpg", img)

    if show:
        cv2.imshow('ArucoServer', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    try:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)  # Detect aruco
    except Exception as e:
        config.log(f"exception in detectMarkers: {e}")

    if ids is not None:  # if aruco marker(s) detected

        return ids, corners

    else:
        return None, None


def calculateMarkerFacts(corners, markerId, camera):
    """
    aruco.estimatePoseSingleMarkers looks to be a bit unprecise, use own calcs for distance and direction
    :param corners:
    :param markerId:
    :param camera:
    :return:
    """

    # marker = markerTypes[markerType=="dockingMarker"]
    marker = [m for m in markerTypes if markerId in m.allowedIds]
    #config.log(f"markerId: {markerId}, markerSize: {marker[0].length}")

    if marker is None or len(marker) == 0:
        return None

    if camera == "CART_CAM":
        colAngle = CARTCAM_X_ANGLE / CARTCAM_X_RESOLUTION
        rowAngle = CARTCAM_Y_ANGLE / CARTCAM_Y_RESOLUTION
        imgXCenter = CARTCAM_X_RESOLUTION / 2
        vec = aruco.estimatePoseSingleMarkers(corners, marker[0].length, config.cartcamMatrix,
                                          config.cartcamDistortionCoeffs)  # For a single marker

    else:
        colAngle = EYECAM_X_ANGLE / EYECAM_X_RESOLUTION
        rowAngle = EYECAM_Y_ANGLE / EYECAM_Y_RESOLUTION
        imgXCenter = EYECAM_X_RESOLUTION / 2
        vec = aruco.estimatePoseSingleMarkers(corners, marker[0].length, config.eyecamMatrix,
                                          config.eyecamDistortionCoeffs)  # For a single marker

    # my markers are not rotated, use average of marker height on left and right side for distance calculation
    # corner indices are clockwise starting with topleft (second index)
    config.log(f"corners: {corners[0]}", publish=False)
    tl=0
    tr=1
    br=2
    bl=3
    col=0
    row=1
    centerCol = (corners[0][tr][col] + corners[0][tl][col]) / 2
    # eval left and right marker height in rows
    markerRowsLeft = corners[0][bl][row] - corners[0][tl][row]
    markerRowsRight = corners[0][br][row] - corners[0][tr][row]
    markerRows = (markerRowsLeft + markerRowsRight) / 2

    # eval the angle of the marker in the image using the vertical cam angle and resolution
    heightAngle = markerRows * rowAngle

    # using the known size of the marker the distance is adj=opp/tan
    # use abs value as eye cam delivers a rotated map
    distanceCamToMarker = abs(marker[0].length / np.tan(np.radians(heightAngle)))

    # use the markers center col to calc the angle to the marker
    angleToMarker = (imgXCenter - centerCol) * colAngle
    #config.log(f"angleToMarker, centerCol: {centerCol}, offset: {imgXCenter - centerCol}, colAngle: {colAngle}")

    # eval the marker's yaw from the result of aruco.estimatePoseSingleMarkers
    rmat = cv2.Rodrigues(vec[0])[0]
    yrp = rotationMatrixToEulerAngles(rmat)

    # markerDegrees is 0 for an orthogonal view position,
    # negative for a viewpoint right of the marker
    # positive for a viewpoint left of the marker
    markerDegrees = float(-np.degrees(yrp[0]))  # ATTENTION, this is the yaw of the marker evaluated from the image

    # for distances > 1500 markerDegrees are not accurate, reduce value
    if distanceCamToMarker > 1500:
        config.log(f"corrected markerDegrees from {markerDegrees} to {markerDegrees/3} because of distance {distanceCamToMarker}")
        markerDegrees = round(markerDegrees / 3)

    log(f"markerId: {markerId}, distanceCamToMarker: {distanceCamToMarker:.0f}, angleToMarker: {angleToMarker:.0f}, markerDegrees: {markerDegrees:.0f}")

    return {'markerId': markerId,
            'distanceCamToMarker': int(distanceCamToMarker),
            'angleToMarker': round(angleToMarker),
            'markerDegrees': round(markerDegrees)}


def calculateCartMovesForDockingPhase1(corners):
    cartCenterPos = point(0, 0)
    camPos = point(0, DISTANCE_CART_CENTER_CAM)

    marker = markerTypes[markerType == "dockingMarker"]

    vec = aruco.estimatePoseSingleMarkers(corners, marker.length, config.cartcamMatrix,
                                          config.cartcamDistortionCoeffs)  # For a single marker

    distanceCamToMarker = vec[1][0, 0, 2]
    xOffsetMarker = vec[1][0, 0, 0]
    log(f"distanceCamToMarker: {distanceCamToMarker:.0f}, xOffsetMarker: {xOffsetMarker:.0f}")
    # log(f"vec[1] {vec[1]}")

    # angle of marker center in cam image, atan of x-offset/distanceCamToMarker
    markerAngleInImage = np.degrees(np.arctan(-xOffsetMarker / distanceCamToMarker))

    # calculate marker position relativ to cart center (cartcam looks straight out)
    markerCenterPos = point(xOffsetMarker, camPos.y + np.cos(np.radians(markerAngleInImage)) * distanceCamToMarker)
    log(
        f"markerAngleInImage: {markerAngleInImage:.1f}, markerCenterPos(relative to cart center): {markerCenterPos.x:.0f} / {markerCenterPos.y:.0f}")

    # eval the marker's yaw (the yaw of the marker itself evaluated from the marker corners)
    rmat = cv2.Rodrigues(vec[0])[0]
    yrp = rotationMatrixToEulerAngles(rmat)
    markerDegrees = -np.degrees(yrp[0])  # ATTENTION, this is the yaw of the marker evaluated from the image

    # orthoAngle = markerAngleInImage + markerDegrees
    orthoAngle = markerDegrees
    xCorr = (marker.orthoStopDist) * np.sin(np.radians(orthoAngle))
    yCorr = (marker.orthoStopDist) * np.cos(np.radians(orthoAngle))
    log(
        f"markerDegrees: {markerDegrees:.1f}, orthoAngle: {orthoAngle:.1f}, xCorr: {xCorr:.0f}, yCorr: {yCorr:.0f}")

    # cart target position is marker's orthogonal point at distance
    # use the offset to account for the x-difference of the docking detail marker center vs the docking marker center
    cartTargetPos = point(markerCenterPos.x + xCorr + MARKER_XOFFSET_CORRECTION, markerCenterPos.y - yCorr)

    log(f"cartTargetPos (cartCenter) = {cartTargetPos.x:.0f},{cartTargetPos.y:.0f}")

    cartStartRotation = np.degrees(np.arctan(cartTargetPos.x / cartTargetPos.y))
    cartMove = np.hypot(cartTargetPos.x, cartTargetPos.y)
    cartEndRotation = -(np.degrees(np.arctan(xCorr / yCorr)) + cartStartRotation)

    return [cartStartRotation, cartMove, cartEndRotation]


def calculateCartMovesForDockingPhase2(corners):
    '''
    cart is expected to be in front of the docking station
    calculate detail moves with the DOCKING_DETAIL_MARKER
    '''

    marker = markerTypes[markerType == "dockingDetail"]

    vec = aruco.estimatePoseSingleMarkers(corners, marker.length, config.cartcamMatrix,
                                          config.cartcamDistortionCoeffs)  # For a single marker

    distanceCamToMarker = vec[1][0, 0, 2]
    xOffsetMarker = vec[1][0, 0, 0]

    # eval the marker's yaw (the yaw of the marker itself evaluated from the marker corners)
    rmat = cv2.Rodrigues(vec[0])[0]
    yrp = rotationMatrixToEulerAngles(rmat)
    markerDegrees = np.degrees(yrp[0])  # ATTENTION, this is the yaw of the marker evaluated from the image

    # rotate only if we are not orthogonal to the marker
    rotation = 0
    if abs(markerDegrees) > 2:
        rotation = markerDegrees

    log(f"rotation: {rotation:.0f}, xOffsetMarker: {xOffsetMarker:.0f}, distanceCamToMarker: {distanceCamToMarker:.0f}")

    return rotation, xOffsetMarker, distanceCamToMarker


def initServer():

    createMarkers = False
    if createMarkers:
        calibration.createMarkers()
        raise SystemExit(0)

    takeCalibrationPicturesEyecam = False
    if takeCalibrationPicturesEyecam:
        path = "C:/Projekte/InMoov/aruco/eyecamCalibration/"
        camID = 0
        time.sleep(10)
        calibration.takeCalibrationPictures(path, camID, -90)
        calibration.createCalibrationMatrixFromImages(path)
        raise SystemExit(0)

    takeCalibrationPicturesCartcam = False
    if takeCalibrationPicturesCartcam:
        path = "C:/Projekte/InMoov/aruco/cartcamCalibration/"
        camID = 1
        #calibration.takeCalibrationPictures(path, camID, 0)
        calibration.createCalibrationMatrixFromImages(path)
        raise SystemExit(0)


    data = np.load("C:/Projekte/InMoov/aruco/cartcamCalibration/calibration.npz")
    config.cartcamMatrix = data['cameraMatrix']
    config.cartcamDistortionCoeffs = data['distortionCoeffs'][0]

    data = np.load("C:/Projekte/InMoov/aruco/eyecamCalibration/calibration.npz")
    config.eyecamMatrix = data['cameraMatrix']
    config.eyecamDistortionCoeffs = data['distortionCoeffs'][0]

    takeEyecamImage()

    # cartcam needs some time to get ready
    for _ in range(3):
        takeCartcamImage()  # cam needs some time to adjust
        time.sleep(0.1)

    config.serverReady = True
    rpcSend.publishServerReady()


if __name__ == "__main__":

    ##########################################################
    # initialization
    # Logging, renaming old logs for reviewing ...
    baseName = f"log/{config.serverName}"
    oldName = f"{baseName}9.log"
    if os.path.isfile(oldName):
        os.remove(oldName)
    for i in reversed(range(9)):
        oldName = f"{baseName}{i}.log"
        newName = f"{baseName}{i+1}.log"
        if os.path.isfile(oldName):
            os.rename(oldName, newName)
    oldName = f"{baseName}.log"
    newName = f"{baseName}0.log"
    if os.path.isfile(oldName):
        try:
            os.rename(oldName, newName)
        except Exception as e:
            config.log(f"can not rename {oldName} to {newName}")

    logging.basicConfig(
        filename=f"log/{config.serverName}.log",
        level=logging.INFO,
        format='%(message)s',
        filemode="w")

    config.log(f"{config.serverName} started")


    windowName = f"marvin//{config.serverName}"
    os.system("title " + windowName)
    # hwnd = win32gui.FindWindow(None, windowName)
    # win32gui.MoveWindow(hwnd, 280,410,1200,400,True)


    if standAloneMode:

        initServer()

        #arucoParams.polygonalApproxAccuracyRate = 0.08
        arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.15     # default = 0.13
        while True:

            file = False
            if file:
                filename = "C:/Users/marvin/Desktop/000_000_300_-30.jpg"
                config.eyecamImg = cv2.imread(filename)
                config.cartcamImg = cv2.imread(filename)

            else:
                config.eyecamImg = takeEyecamImage()
                config.cartcamImg = takeCartcamImage()

            # CARTCAM
            if config.cartcamImg is not None:
                cv2.imshow("cartCam", config.cartcamImg)

                ids, corners = findMarkers(config.cartcamImg, False)
                if ids is not None:
                    print(ids)
                    for markerIndex, foundId in enumerate(ids):
                        markerInfo = calculateMarkerFacts(corners[markerIndex], foundId, "CART_CAM")
                        #config.log(f"corners: {corners}")
                        config.log(f"markerInfo: {markerInfo}")
                else:
                    log(f"no marker found")

                cv2.waitKey(0)


            # EYECAM
            if config.eyecamImg is not None:
                cv2.imshow("eyeCam", config.eyecamImg)

                ids, corners = findMarkers(config.eyecamImg, False)

                if ids is not None:
                    print(ids)
                    for markerIndex, foundId in enumerate(ids):
                        markerInfo = calculateMarkerFacts(corners[markerIndex], foundId, "EYE_CAM")
                else:
                    log(f"no marker found")

                cv2.waitKey(0)
                cv2.destroyAllWindows()

            raise SystemExit(0)

    else:

        # initialize cams
        initThread = threading.Thread(target=initServer, args={})
        initThread.setName("initServer")
        initThread.start()

        # start the watchDog for clientConnections
        navThread = threading.Thread(target=watchDog.watchDog, args={})
        navThread.setName("connectionWatchDog")
        navThread.start()

        print(f"start listening on port {config.MY_RPC_PORT}")
        myConfig = {"allow_all_attrs": True, "allow_pickle": True}
        listener = ThreadedServer(rpcReceive.arucoListener, port=config.MY_RPC_PORT, protocol_config=myConfig)

        listener.start()




