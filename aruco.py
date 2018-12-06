import glob
import os
import time

import cv2
import imutils
import numpy as np
from cv2 import aruco
from pyrecord import Record
import threading
from rpyc.utils.server import ThreadedServer

import config
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
markerTypes = [markerType('dockingMarker', 80, 1, 600, [10]),
               markerType('dockingDetail', 60, 1, 0, [11]),
               markerType('objectMarker', 10, 1, 600, [2])]


# ARUCO_PORT = 20001
DOCKING_MARKER_ID = 10
DOCKING_DETAIL_ID = 11

DISTANCE_CART_CENTER_CAM = 330  # mm
MARKER_XOFFSET_CORRECTION = -22

eyecam = None
_eyecamImg = None

_markerId = 0
cartcam = None
_cartcamImg = None

# calibrateCamera
# createMarkers()
# calibration.createCharucoBoard()
# calibration.takeCalibrationPictures()
# calibration.calibrateCamera()

# calibration.calibrateCameraCharuco()
# exit()

InMoovEyeCamMatrix = None
InMoovEyeCamDistortionCoeffs = None

ps3eyeMatrix = None
ps3eyeDistortionCoeffs = None

arucoParams = aruco.DetectorParameters_create()
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

navManager = None
cap = None
lastImg = None


def log(msg):
    if navManager is not None:
        navManager.root.recordLog("aruco - " + msg)
    print(msg)


def arucoInit():
    global eyecam, cartcam

    # try to capture an image with the eyecam
    # try:
    eyecam = cv2.VideoCapture(0)
    if eyecam is None:
        log(f"eyecam could not connect to VideoCapture(0)")
        return False

    for i in range(10):
        time.sleep(0.2)
        _, _ = eyecam.read()
        ret, eyecamImg = eyecam.read()
        if not ret:
            log(f"eyecam could not aquire image")
            eyecam.release()
            return False

        if eyecamImg.max() == 0:  # test for all black pix
            continue

        log(f"eyecam image successfully aquired {eyecam}")
        if standAloneMode:
            cv2.imshow("eyecamImg", imutils.rotate_bound(eyecamImg, 90))
            cv2.waitKey()
            cv2.destroyAllWindows()
        break

    # try to capture an image with the cartcam
    cartcam = cv2.VideoCapture(1)
    if cartcam is None:
        log(f"cartcam, could not connect to VideoCapture(1)")
        return False

    for i in range(10):
        time.sleep(0.2)

        _, _ = cartcam.read()
        ret, cartcamImg = cartcam.read()

        if not ret:
            log(f"cartcam, could not aquire image")
            cartcam.release()
            return False

        if cartcamImg.max() == 0:  # ignore black only images
            continue

        log(f"cartcam image successfully aquired {cartcam}")
        if standAloneMode:
            cv2.imshow("cartcamImg", imutils.rotate_bound(cartcamImg, 180))
            cv2.waitKey()
            cv2.destroyAllWindows()
        break

    # remove previous runs images
    for f in glob.glob(f"images/*.jpg"):
        os.remove(f)

    return True


def lookForMarkers(camera, markerIds):
    foundMarkers = []

    if camera == "EYE_CAM":
        img = _eyecamImg

    elif camera == "CART_CAM":
        img = _cartcamImg

    else:
        config.log(f"requested camera {camera} not handled in call")
        return []

    # check for marker
    foundIds, corners = findMarkers(img, show=False)

    if len(markerIds) > 0 and foundIds is None:
        config.log(f"none of the markers {markerIds} found in image")
        return []

    if foundIds is not None:
        config.log(f"markers found: {foundIds}")
        for markerIndex, foundId in enumerate(foundIds):
            if len(markerIds) == 0 or foundId in markerIds:
                markerInfo = calculateImageFacts(corners[markerIndex], foundId)
                foundMarkers.append(markerInfo)

    return foundMarkers


def getEyecamImage():
    global _eyecamImg, eyecam

    config.log(f"eyecam.read() {eyecam}")
    if eyecam is None:
        eyecam = cv2.VideoCapture(0)
    if eyecam is None:
        config.log(f"could not read from VideoCapture(0)")
        return None

    _, _ = eyecam.read()
    ret, _eyecamImg = eyecam.read()
    if ret:
        config.log("eyecam image successfully taken")
        _eyecamImg = imutils.rotate_bound(_eyecamImg, 90)
        return _eyecamImg

    else:
        config.log("error taking eyecam image")
        return None


def getCartcamImage():
    global _cartcamImg, cartcam

    config.log(f"cartcam.read() {cartcam}")
    if cartcam is None:
        cartcam = cv2.VideoCapture(1)
    if cartcam is None:
        config.log(f"could not read from VideoCapture(1)")

    _, _ = cartcam.read()
    ret, _cartcamImg = cartcam.read()
    if ret:
        _cartcamImg = imutils.rotate_bound(_cartcamImg, 180)
        config.log("cartcam image successfully taken")
        return _cartcamImg
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
def evalYawDistToCartTarget(yawToMarker, distance, markerYaw):
    log(
        f"evalYawDistToCartDestination yawToMarker: {yawToMarker:.0f}, distance: {distance:.0f}, markerYaw: {markerYaw:.0f}")

    # Position x,y of camera
    p1 = point(0, 0)

    # Center of Marker
    p2 = point(0, 0)
    p2.x = int(distance * np.cos(np.radians(yawToMarker)))
    p2.y = int(distance * np.sin(np.radians(yawToMarker)))
    # print(p2)

    # angle between marker and cart destination point (includes markerYaw)
    beta = yawToMarker - 90 + markerYaw

    # cart destination point orthogonal in front of marker with offset
    p3 = point(0, 0)
    p3.x = int(p2.x + (distance * np.sin(np.radians(beta))))
    p3.y = int(p2.y - (distance * np.cos(np.radians(beta))))
    # print(p3)

    # angle to cart point in relation to yawToMarker
    yawToCartTarget = 180 + np.degrees(np.arctan(p3.y / p3.x))
    distToCartTarget = np.sqrt(np.power(p3.x, 2) + np.power(p3.y, 2))

    log(
        f"yawToCartTarget: {yawToCartTarget:.0f}, distToCartTarget {distToCartTarget:.0f}, markerPos: {p2}, cartTargetAngle: {beta:.0f}, cartTargetPos: {p3}")

    return yawToCartTarget, distToCartTarget


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

    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)  # Detect aruco

    if ids is not None:  # if aruco marker(s) detected

        return ids, corners

    else:
        return None, None


def calculateImageFacts(corners, markerId):
    # marker = markerTypes[markerType=="dockingMarker"]
    marker = [m for m in markerTypes if markerId in m.allowedIds]

    vec = aruco.estimatePoseSingleMarkers(corners, marker[0].length, ps3eyeMatrix,
                                          ps3eyeDistortionCoeffs)  # For a single marker

    distanceCamToMarker = vec[1][0, 0, 2]
    xOffsetMarker = vec[1][0, 0, 0]
    angleToMarker = np.degrees(np.arctan(-xOffsetMarker/distanceCamToMarker))

    # eval the marker's yaw (the yaw of the marker itself evaluated from the marker corners)
    rmat = cv2.Rodrigues(vec[0])[0]
    yrp = rotationMatrixToEulerAngles(rmat)
    markerOrientation = -np.degrees(yrp[0])  # ATTENTION, this is the yaw of the marker evaluated from the image

    log(
        f"markerId: {markerId}, distanceCamToMarker: {distanceCamToMarker:.0f}, xOffsetMarker: {xOffsetMarker:.0f}, markerOrientation: {markerOrientation:.0f}")

    return {'markerId': markerId,
            'distanceCamToMarker': distanceCamToMarker,
            'angleToMarker': angleToMarker,
            'markerOrientation': markerOrientation}


def calculateCartMovesForDockingPhase1(corners):
    cartCenterPos = point(0, 0)
    camPos = point(0, DISTANCE_CART_CENTER_CAM)

    marker = markerTypes[markerType == "dockingMarker"]

    vec = aruco.estimatePoseSingleMarkers(corners, marker.length, ps3eyeMatrix,
                                          ps3eyeDistortionCoeffs)  # For a single marker

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
    markerYawDegrees = -np.degrees(yrp[0])  # ATTENTION, this is the yaw of the marker evaluated from the image

    # orthoAngle = markerAngleInImage + markerYawDegrees
    orthoAngle = markerYawDegrees
    xCorr = (marker.orthoStopDist) * np.sin(np.radians(orthoAngle))
    yCorr = (marker.orthoStopDist) * np.cos(np.radians(orthoAngle))
    log(
        f"markerYaw[degrees]: {markerYawDegrees:.1f}, orthoAngle: {orthoAngle:.1f}, xCorr: {xCorr:.0f}, yCorr: {yCorr:.0f}")

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

    vec = aruco.estimatePoseSingleMarkers(corners, marker.length, ps3eyeMatrix,
                                          ps3eyeDistortionCoeffs)  # For a single marker

    distanceCamToMarker = vec[1][0, 0, 2]
    xOffsetMarker = vec[1][0, 0, 0]

    # eval the marker's yaw (the yaw of the marker itself evaluated from the marker corners)
    rmat = cv2.Rodrigues(vec[0])[0]
    yrp = rotationMatrixToEulerAngles(rmat)
    markerYawDegrees = np.degrees(yrp[0])  # ATTENTION, this is the yaw of the marker evaluated from the image

    # rotate only if we are not orthogonal to the marker
    rotation = 0
    if abs(markerYawDegrees) > 2:
        rotation = markerYawDegrees

    log(f"rotation: {rotation:.0f}, xOffsetMarker: {xOffsetMarker:.0f}, distanceCamToMarker: {distanceCamToMarker:.0f}")

    return rotation, xOffsetMarker, distanceCamToMarker


'''
def checkForTasks():

    global _eyecamImg, _cartcamImg, _takeEyecamImg, _takeCartcamImg

    while True:

        try:   
            if _takeEyecamImg:

                ret, _eyecamImg = eyecam.read()
                ret, _eyecamImg = eyecam.read()
                _takeEyecamImage = False

                if not ret:
                    log(f"capture image failed, imgId: {eyecamImgId}")

                _eyecamImg = rotatedImg(_eyecamImg, 270)

                # send picture to navManager
                navManager.root.eyecamImgReady(_eyecamImgId, _eyecamImg)

                # then check for marker
                ids, corners = findMarker(_eyecamImg, False)

                # and send result to navManager
                # TODO


        except EOFError as error:
            print("EOF Error in checkForTasks, try to restart the aruco task")
            print()
            print("!!!!!!!!!")
            print("PLEASE RESTART MANUALLY, programmatical restart does not work because the cams do not get released")
            print("!!!!!!!!!")

            raise SystemExit(0)

'''

if __name__ == "__main__":

    windowName = "marvin//aruco"
    os.system("title " + windowName)
    # hwnd = win32gui.FindWindow(None, windowName)
    # win32gui.MoveWindow(hwnd, 280,410,1200,400,True)

    path = "../ps3eye/"
    camID = 1

    createMarkers = False
    if createMarkers:
        calibration.createMarkers()
        raise SystemExit(0)

    takeCalibrationPictures = False
    if takeCalibrationPictures:
        calibration.takeCalibrationPictures(path, camID, 0)
        raise SystemExit(0)

    createCalibrationFile = False
    if createCalibrationFile:
        calibration.createCalibrationMatrixFromImages(path)
        raise SystemExit(0)

    data = np.load("ps3eye/calibration.npz")
    ps3eyeMatrix = data['cameraMatrix']
    ps3eyeDistortionCoeffs = data['distortionCoeffs'][0]

    data = np.load("InMoovEyeCam/calibration.npz")
    InMoovEyecamMatrix = data['cameraMatrix']
    InMoovEyeCamDistortionCoeffs = data['distortionCoeffs'][0]

    if not arucoInit():
        os._exit(1001)

    if standAloneMode:

        while True:
            _, _ = cartcam.read()
            ret, _cartcamImg = cartcam.read()
            if ret:
                _cartcamImg = imutils.rotate_bound(_cartcamImg, 180)
            else:
                log("error in capturing cartcam image")
                raise SystemExit(0)

            ids, corners = findMarkers(_cartcamImg, False)
            if ids is not None:
                if DOCKING_MARKER_ID in ids:
                    itemIndex = np.where(ids == DOCKING_MARKER_ID)
                    startRotation, cartMove, endRotation = calculateCartMovesForDockingPhase1(corners[itemIndex[0][0]])

                    log(
                        f"send startDockingPhase1 to navManager, startRotation: {startRotation:.0f}, cartMove: {cartMove:.0f}, endRotation: {endRotation:.0f}")

            cv2.imshow("docking station", _cartcamImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

    else:

        # start the watchDog for clientConnections
        navThread = threading.Thread(target=watchDog.watchDog, args={})
        navThread.setName("connectionWatchDog")
        navThread.start()

        print(f"start listening on port {config.MY_RPC_PORT}")
        myConfig = {"allow_all_attrs": True, "allow_pickle": True}
        server = ThreadedServer(rpcReceive.arucoListener, port=config.MY_RPC_PORT, protocol_config=myConfig)

        server.start()




