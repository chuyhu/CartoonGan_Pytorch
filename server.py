import json
import socket

from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket
from thrift.transport import TTransport

from contacts.APITransmit import APITransmit


class APITransmitHandler:
    def __init__(self):
        self.log = {}

    def sayMsg(self, msg):
        msg = json.loads(msg)
        print("sayMsg(" + msg + ")")
        return "say " + msg + " from " + socket.gethostbyname(socket.gethostname())

    def invoke(self, cmd, token, data):
        cmd = cmd
        token = token
        data = data
        if cmd == 1:
            return json.dumps({token: data})
        else:
            return 'cmd不匹配'


class APIServer:

    def __init__(self, address, useUnixSocket=False):
        if useUnixSocket:
            self.transport = TSocket.TServerSocket(unix_socket=address)
        else:
            self.transport = TSocket.TServerSocket(host=address[0], port=address[1])

        self.tfactory = TTransport.TBufferedTransportFactory()
        self.pfactory = TBinaryProtocol.TBinaryProtocolFactory()
        self.handler = APITransmitHandler()
        self.processor = APITransmit.Processor(self.handler)
        self.server = TServer.TSimpleServer(self.processor, self.transport, self.tfactory, self.pfactory)

    def start(self):
        self.server.serve()
        pass

    def close(self):
        self.transport.close()
        pass
