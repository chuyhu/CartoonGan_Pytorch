import socket
import logging
import asyncio


class SocketServer:
    def __init__(self, address, useUnixSocket=False, backlog=50):
        self.backlog = backlog
        self.address = address
        self.useUnixSocket = useUnixSocket
        self.shouldStop = False
        if self.useUnixSocket:
            self.s = socket.socket(1, socket.SOCK_STREAM)
        else:
            self.s = socket.socket()
        pass

    async def start(self, onAccept):
        logging.debug("Start socket server on %s" % str(self.address))
        self.s.bind(self.address)
        self.s.listen(self.backlog)
        loop = asyncio.get_event_loop()

        while self.shouldStop is False:
            conn = self.s.accept()
            loop.create_task(onAccept(conn))
        pass

    async def close(self):
        self.shouldStop = True
        self.s.close()
        pass

    pass
