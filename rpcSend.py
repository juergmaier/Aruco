
import config
import rpcReceive


def publishLog(msg):

    for i, c in enumerate(rpcReceive.clientList):

        if c['replyConn'] is not None:

            try:
                c['replyConn'].root.exposed_log(msg)

            except Exception as e:
                print(f"exception in publishLog with {c['clientId']}: {e}")
                c['replyConn'] = None

