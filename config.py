

import datetime
import logging
import rpcSend

clientList = []

MY_IP = "192.168.0.17"
MY_RPC_PORT = 20002      # defined in taskOrchestrator
serverName = 'aruco'
serverReady = False

eyecamMatrix = None
eyecamDistortionCoeffs = None

cartcamMatrix = None
cartcamDistortionCoeffs = None

eyecam = None
eyecamImg = None

cartcam = None
cartcamImg = None

_markerId = 0


def log(msg, publish=True):
    logtime = str(datetime.datetime.now())[11:]
    print(f"{logtime} - {msg}")
    logging.info(f"{logtime} - {msg}")

    if publish:
        rpcSend.publishLog(f"{serverName} - " + msg)

