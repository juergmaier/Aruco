
import os
import time
import rpyc

import config
import aruco

clientList = []
watchInterval = 5

server = 'aruco'


class arucoListener(rpyc.Service):

    ############################## common routines for servers
    def on_connect(self, conn):
        print(f"on_connect in server seen {conn}")
        callerName = conn._channel.stream.sock.getpeername()
        print(f"caller: {callerName}")

    def on_disconnect(self, conn):
        callerName = conn._channel.stream.sock.getpeername()
        print(f"{server} - on_disconnect triggered, conn: {callerName}")

    def exposed_requestForReplyConnection(self, ip, port, interval=5):

        print(f"request for reply connection {ip}:{port}")
        replyConn = rpyc.connect(ip, port)

        clientId = (ip, port)
        connectionUpdated = False
        for c in clientList:
            if c['clientId'] == clientId:
                print(f"update client connection")
                c['replyConn'] = replyConn
                connectionUpdated = True

        if not connectionUpdated:
            print(f"append client connection {clientId}")
            clientList.append({'clientId': clientId, 'replyConn': replyConn, 'lastMessageReceivedTime': time.time(), 'interval': interval})


    def exposed_getLifeSignal(self, ip, port):

        for c in clientList:
            if c['clientId'] == (ip, port):
                #print(f"life signal received from  {ip}, {port}, {time.time()}")
                c['lastMessageReceivedTime'] = time.time()

        return True

    def exposed_terminate(self):
        print(f"{server} task - terminate request received")
        os._exit(0)
        return True
    ############################## common routines for servers


    def exposed_findMarkers(self, camera, markerIds):
        return aruco.lookForMarkers(camera, markerIds)

    def exposed_getEyecamImage(self):
        #img = aruco.getEyecamImage()
        #transferImg = img.tostring()
        #return transferImg
        return aruco.getEyecamImage()

    def exposed_getCartcamImage(self):
        #img = aruco.getCartcamImage()
        #transferImg = img.tostring()
        #return transferImg
        return aruco.getCartcamImage()

    '''
    def exposed_findDockingDetailMarker(self, markerId):

        config.logf"findDockingDetailMarker received")
        ret, _cartcamImg = cartcam.read()
        ret, _cartcamImg = cartcam.read()

        if not ret:
            config.logf"capture image cartcam failed, imgId: {_cartcamImgId}")
            return [False, None, None, None]

        _cartcamImg = rotatedImg(_cartcamImg, 180)

        cv2.imwrite("dockingDetailMarker.jpg", _cartcamImg)

        # then check for marker
        ids, corners = findMarker(_cartcamImg, False)

        if ids is None:
            config.logf"no marker in image")
            return [False, None, None, None]
        else:
            config.logf"markers found: {ids}")

        # sequence for docking phase 2 position approach
        if DOCKING_DETAIL_ID in ids:
            itemIndex = np.where(ids[0] == DOCKING_DETAIL_ID)
            rotation, xOffset, distance = calculateCartMovesForDockingPhase2(corners[itemIndex[0][0]])
            return [True, rotation, xOffset + MARKER_XOFFSET_CORRECTION, distance]

        else:
            config.log"docking detail marker not found")
            return [False, None, None, None]
    '''

