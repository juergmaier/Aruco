

import datetime

import rpcSend


MY_IP = "192.168.0.17"
MY_RPC_PORT = 20002      # defined in taskOrchestrator

def log(msg):
    logtime = str(datetime.datetime.now())[11:]
    print(f"{logtime} - {msg}")

    rpcSend.publishLog("aruco - " + msg)

