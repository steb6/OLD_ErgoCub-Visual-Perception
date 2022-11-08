from abc import abstractmethod

from utils.concurrency.yarpnode import _exception_handler, YarpNode
from loguru import logger


class SrcYarpNode(YarpNode):

    def _startup(self):
        logger.info('Starting up...')
        self.startup()
        logger.success('Start up complete.')

    @abstractmethod
    def loop(self) -> dict:
        pass

    # @_exception_handler
    # @logger.catch(reraise=True)
    def run(self) -> None:
        self._startup()

        # This is to wait for the first message even in non-blocking mode
        while True:
            data = self.loop()
            data = self.prepare(data)
            self._send_all(data, self.blocking)
