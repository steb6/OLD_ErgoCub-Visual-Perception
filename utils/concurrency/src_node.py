from abc import abstractmethod

from utils.concurrency.node import _exception_handler, Node
from loguru import logger


class SrcNode(Node):

    def _startup(self):
        logger.info('Starting up...')
        self.startup()
        logger.success('Start up complete.')

    @abstractmethod
    def loop(self) -> dict:
        pass

    @_exception_handler
    @logger.catch(reraise=True)
    def run(self) -> None:
        self._startup()

        # This is to wait for the first message even in non-blocking mode
        while True:
            data = self.loop()
            self._send_all(data, self.blocking)
