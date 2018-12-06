import glob
import time
import numpy as np
import cv2
from cv2 import aruco


def createMarkers():
    for i in range(0,20):
        aruco_dict = aruco.Dictionary_get( aruco.DICT_5X5_50 )
        img = aruco.drawMarker(aruco_dict, i, 500)
        cv2.imwrite("marker" + str(i) + ".jpg", img)

def createCharucoBoard():
    aruco_dict = aruco.Dictionary_get( aruco.DICT_5X5_50 )
    board = aruco.CharucoBoard_create(6, 8, 0.025, 0.02, aruco_dict )
    img = board.draw((200*3,200*3))
    cv2.imwrite("CharucoBoard.jpg", img)

def takeCalibrationPictures(path, camId, rotation):

    arucoParams = aruco.DetectorParameters_create()
    arucoDict = aruco.Dictionary_get( aruco.DICT_5X5_50 )
    board = aruco.CharucoBoard_create(6, 8, 0.025, 0.02, arucoDict )

    cap = cv2.VideoCapture(camId)
    time.sleep(1)
    if not cap.isOpened():
        cap.open()

    i=0
    while i < 20:

        ret, img = cap.read()

        if ret:
            rows,cols = img.shape[0:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            cv2.imshow('charuco', img)
            cv2.waitKey(1000)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #ret, corners = cv2.findChessboardCorners(gray, (5,6),None)

            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams) # Detect aruco
            #aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)
       

            #print("# ids: " + str(len(ids)))
            if ids is not None:
                i += 1
                filename = path + "calibrationImage_" + str(i) + ".jpg"
                cv2.imwrite(filename, img)
                print("image " + filename + " saved")
            else:
                print(ids)
        else:
            print("could not take picture")

'''
def createCalibrationMatrixFromImages():

    arucoParams = aruco.DetectorParameters_create()
    aruco_dict = aruco.Dictionary_get( aruco.DICT_5X5_50 )
    board = aruco.CharucoBoard_create(4, 5, 5, 4, aruco_dict )

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((4*5,3), np.float32)
    objp[:,:2] = np.mgrid[0:5,0:4].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    images = glob.glob('../chessboard/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (4,5),None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (6,5), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(2000)

    
    # calculate camera distortion
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez("../ps3eye/calibration", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    print("camera matrix:\n", mtx)
    print("distortion coefficients: ", dist.ravel())
'''

def createCalibrationMatrixFromImages(path):

    arucoParams = aruco.DetectorParameters_create()
    arucoDict = aruco.Dictionary_get( aruco.DICT_5X5_50 )
    board = aruco.CharucoBoard_create(6, 8, 0.025, 0.02, arucoDict )

    allCorners = []
    allIds = []

    images = glob.glob(path + '*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
        res = cv2.aruco.detectMarkers(gray,arucoDict)

        if len(res[0])>0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3:
                allCorners.append(res2[1])
                allIds.append(res2[2])

            cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    imsize = gray.shape


    try:
        cal = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
        np.savez(path + "calibration", cameraMatrix=cal[1], distortionCoeffs=cal[2])
        print(f"calibration data saved, path: {path}")
        print(cal[1])
        print(cal[2])
    except:
        print("no calibration possible with provided images")

    cv2.destroyAllWindows()
