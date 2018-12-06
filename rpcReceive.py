
import os
import time
import rpyc

import config
import aruco

clientList = []
watchInterval = 5

server = 'aruco'


def addClient(c, i):

    global clientList, watchInterval

    clientList.append(c.copy())
    watchInterval = i
    print(f"client added, {c}")


def updateMessageTime(pid):

    global clientList

    for i, c in enumerate(clientList):
        if c['pid'] == pid:
            #print(f"{time.time():.0f} update message time, pid {pid}, clientIndex: {i}")
            clientList[i]['lastMessageReceivedTime'] = time.time()
            #config.log(f"getLifeSignal time update, pid: {pid}, clientIndex: {i}, time: {time.time()}")


def removeClient(i):

    global clientList

    print(f"remove client from clientList, index: {i}, client: {clientList[i]}")
    del clientList[i]



class arucoListener(rpyc.Service):

    ############################## common routines for clients
    watchInterval = 5

    def on_connect(self, conn):

        print(f"{server} - on_connect triggered")
        callerName = conn._channel.stream.sock.getpeername()
        #self.persistConn = conn
        clientPid, clientInterval = conn.root.exposed_getPid()

        if clientInterval < self.watchInterval:   # use shortest client interval in watchdog loop
            self.watchInterval = clientInterval

        clientIndex = [i for i in clientList if i['conn'] == conn]
        if len(clientIndex) == 0:
            client = {'conn': conn,
                      'callerName': callerName,
                      'pid': clientPid,
                      'interval': clientInterval,
                      'lastMessageReceivedTime': time.time()}
            addClient(client, self.watchInterval)
        #config.log(f"on_connect in '{server}' with {client}")


    def on_disconnect(self, conn):
        callerName = conn._channel.stream.sock.getpeername()
        print(f"{server} - on_disconnect triggered, conn: {callerName}")


    def exposed_getLifeSignal(self, pid):

        updateMessageTime(pid)
        return True


    def exposed_terminate(self):
        print(f"{server} task - terminate request received")
        os._exit(0)
        return True

    ############################## common routines for clients


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

