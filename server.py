import commonserver


class APIServer:

    def __init__(self, address, useUnixSocket=False, backlog=50):
        self.server = commonserver.SocketServer(address, useUnixSocket, backlog)

    async def start(self):
        await self.server.start(self)
        pass

    async def close(self):
        await self.server.close()
        pass
