
import time

import config
import rpcReceive

# check for life signal from client while having a connection

def watchDog():
    while True:

        # watchdog for incoming commands from clients
        # stop cart if we do not get regular messages
        for i, c in enumerate(rpcReceive.clientList):

            if c['replyConn'] is not None:

                span = (time.time() - c['lastMessageReceivedTime'])
                if (span) > (c['interval'] * 1.5):
                    config.log(
                        f"{time.time():.0f} heartbeat interval exceeded, span: {span:.0f}, interval: {c['interval']}")
                    c['replyConn'] = None

        # time.sleep(rpcReceive.watchInterval)
        time.sleep(rpcReceive.watchInterval)

