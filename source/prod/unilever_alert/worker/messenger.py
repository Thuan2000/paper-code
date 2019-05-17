import json
import imageio
from socketIO_client_nexus import SocketIO, BaseNamespace
import prod.unilever_alert.config as Config


class SocketMessenger():

    def __init__(self, hostname, port):
        imageio.plugins.freeimage.download()
        self.hostname = hostname
        self.port = port
        self.socket = SocketIO(self.hostname, self.port)
        self.messenger_namespace = self.socket.define(MessengerNameSpace,
            Config.Socket.NAMESPACE)

    def send_alert(self, **kwargs):
        json_string = json.dumps(kwargs)
        self.messenger_namespace.emit(Config.Socket.ALERT_EVENT, json_string)

    def send_statistic(self, **kwargs):
        json_string = json.dumps(kwargs)
        print(json_string)
        self.messenger_namespace.emit(Config.Socket.STATISTIC, json_string)


class MessengerNameSpace(BaseNamespace):

    def on_connect(self, *args):
        print('Connecting...', args)
