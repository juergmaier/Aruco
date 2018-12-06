
import time

import config
import rpcReceive

# check for life signal from client while having a connection

def watchDog():

    while True:

        # watchdog for incoming commands from navManager
        # stop cart if we do not get regular messages
        for i, c in enumerate(rpcReceive.clientList):
            if (time.time() - c['lastMessageReceivedTime']) > (c['interval'] * 1.5):

                config.log(f"{time.time():.0f} heartbeat interval exceeded, {c['lastMessageReceivedTime']:.0f}, removeClient")
                rpcReceive.removeClient(i)

    time.sleep(rpcReceive.watchInterval)