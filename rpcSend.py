
import config


def publishLog(msg):

    for i, c in enumerate(config.clientList):

        if c['replyConn'] is not None:

            try:
                c['replyConn'].root.exposed_log(msg)

            except Exception as e:
                print(f"exception in publishLog with {c['clientId']}: {e}")
                c['replyConn'] = None



def publishLifeSignal():

    #config.log(f"publishing life signal")
    for i, c in enumerate(config.clientList):

        if c['replyConn'] is not None:
            try:
                c['replyConn'].root.exposed_lifeSignalUpdate(config.serverName)
            except Exception as e:
                c['replyConn'] = None
                config.log(f"exception in publishLifeSignal with {c['clientId']}: {e}")


def publishServerReady():

    config.log(f"publishing ready={config.serverReady} message to clients")

    for i, c in enumerate(config.clientList):

        if c['replyConn'] is not None:
            if len(c['messageList']) == 0 or 'serverReady' in c['messageList']:
                try:
                    c['replyConn'].root.exposed_serverReady(config.serverName, config.serverReady)
                except Exception as e:
                    c['replyConn'] = None
                    config.log(f"exception in publishServerReady with {c['clientId']}: {e}")

