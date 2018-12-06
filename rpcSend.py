
import config
import rpcReceive


def publishLog(msg):

    for i, c in enumerate(rpcReceive.clientList):
        #print(f"log to {xmlrpcClientList[i]}")
        try:
            c['conn'].root.exposed_log(msg)

        except Exception as e:
            print(f"exception in publishLog with c.callerName: {str(e)}")

