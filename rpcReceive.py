
import os
import time
import rpyc

import config
import aruco
import rpcSend


watchInterval = 5



class arucoListener(rpyc.Service):

    ############################## common routines for servers
    def on_connect(self, conn):
        print(f"on_connect in server seen {conn}")
        callerName = "unknown"
        try:
            callerName = conn._channel.stream.sock.getpeername()
        except Exception as e:
            config.log(f"on_connect, could not get peername: {e}")
        print(f"caller: {callerName}")


    def on_disconnect(self, conn):
        callerName = "unknown"
        try:
            callerName = conn._channel.stream.sock.getpeername()
        except Exception as e:
            config.log(f"on_disconnect, could not get peername: {e}")
        print(f"{config.serverName} - on_disconnect triggered, conn: {callerName}")


    def exposed_requestForReplyConnection(self, ip, port, messages=[], interval=5):

        config.log(f"request for reply connection received from {ip}:{port}, messageList: {messages}")

        try:
            replyConn = rpyc.connect(ip, port)
            config.log(f"reply connection established")
        except Exception as e:
            config.log(f"failed to open a reply connection, {e}")
            return

        clientId = (ip, port)
        connectionUpdated = False
        for c in config.clientList:
            if c['clientId'] == clientId:
                print(f"update client connection")
                c['replyConn'] = replyConn
                c['lastMessageReceivedTime'] = time.time()
                c['messageList'] = list(messages)
                c['interval'] = interval
                connectionUpdated = True

        if not connectionUpdated:
            config.log(f"append client connection {clientId}")
            config.clientList.append({'clientId': clientId,
                                      'replyConn': replyConn,
                                      'lastMessageReceivedTime': time.time(),
                                      'messageList': list(messages),
                                      'interval': interval})

        # if server is already running send a ready message
        if config.serverReady:
            rpcSend.publishServerReady()
        else:
            rpcSend.publishLifeSignal()


    def exposed_requestLifeSignal(self, ip, port):

        #config.log(f"life signal request received")
        for c in config.clientList:
            if c['clientId'] == (ip, port):
                #print(f"life signal received from  {ip}, {port}, {time.time()}")
                c['lastMessageReceivedTime'] = time.time()
        rpcSend.publishLifeSignal()


    def exposed_terminate(self):
        print(f"{config.serverName} task - terminate request received")
        os._exit(0)
        return True

    ############################## end common routines for servers

    def exposed_findMarkers(self, camera, markerIds, camYaw):
        config.log(f"findMarkers request received", publish=False)
        return aruco.lookForMarkers(camera, markerIds, camYaw)


    def exposed_getEyecamImage(self):
        return aruco.takeEyecamImage()


    def exposed_getCartcamImage(self):
        config.log(f"getCartCamImage request received", publish=False)
        return aruco.takeCartcamImage()


