import time
from abc import ABC, abstractmethod
from multiprocessing import Process

from queue import Empty, Full

import numpy as np
from loguru import logger
import yarp


def _exception_handler(function):
    # Graceful shutdown
    def wrapper(*args):
        try:
            function(*args)
        except Exception as e:
            logger.exception(e)
            exit(1)

    return wrapper


def connect(manager):
    logger.info('Connecting to manager...')
    start = time.time()

    while True:
        try:
            manager.connect()
            break
        except ConnectionRefusedError as e:
            if time.time() - start > 120:
                logger.error('Connection refused.')
                raise e
            time.sleep(1)
    logger.success('Connected to manager.')


class YarpNode(Process, ABC):

    def __init__(self, in_queue=None, out_queues=None, blocking=False):
        super(Process, self).__init__()

        self.blocking = blocking

        yarp.Network.init()

        if in_queue is not None:
            in_queue = '/' + in_queue + '_in'
            self._in_queue = yarp.BufferedPortBottle()
            self._in_queue.open("in_queue")

        self._out_queues = {}
        for out_q in out_queues:
            p = yarp.BufferedPortBottle()
            p.open('/' + out_q + '_out')
            yarp.Network.connect('/' + out_q + '_out', '/' + out_q + '_in')

            self._out_queues[out_q] = p


        logger.info(f'Input queue: {in_queue} - Output queues: {", ".join(out_queues)}')

    def _startup(self):
        logger.info('Starting up...')
        self.startup()
        logger.info('Waiting for source startup...')
        data = self._recv()
        logger.success('Start up complete.')

    def _recv(self):

        data = self._in_queue.get()

        return data

    def _recv_nowait(self):
        if not self._in_queue.empty():
            return self._recv()
        else:
            return None

    def _send_all(self, data, blocking):
        for dest in data:
            self._out_queues[dest].write()

    def startup(self):
        pass

    def shutdown(self):
        pass

    # @abstractmethod
    # def loop(self, data: dict) -> dict:
    #     pass

    # @_exception_handler
    # @logger.catch(reraise=True)
    def run(self) -> None:
        self._startup()

        # This is to wait for the first message even in non-blocking mode
        data = self._recv()
        data = self.unpack(data)
        while True:
            data = self.loop(data)
            data = self.prepare(data)
            self._send_all(data, self.blocking)

            data = self._recv()

    def prepare(self, data):
        for dest in data:
            bottle = self._out_queues[dest].prepare()
            bottle.clear()
            for k, v in data[dest].items():
                if isinstance(v, np.ndarray):
                    v = v.astype(np.float32)
                    yarp_data = yarp.ImageRgb()
                    yarp_data.setExternal(v, v.shape[1], v.shape[0])

                property = bottle.addDict()
                property.put(k, yarp.Value(yarp_data, 10))
        return data

    def unpack(self, data):
        data.find()


if __name__ == '__main__':
    YarpNode('source', ['sink', 'action_recognition'])
