import time
from abc import ABC, abstractmethod
from multiprocessing import Process
from multiprocessing.managers import BaseManager

from loguru import logger
from queue import Empty


def _exception_handler(function):
    def wrapper(*args):
        try:
            function(*args)
        except:
            logger.exception('Exception Raised')
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


class Node(Process, ABC):

    def __init__(self, ip, port, in_queue=None, out_queues=None, blocking=False):
        super(Process, self).__init__()

        self.blocking = blocking

        BaseManager.register('get_queue')
        manager = BaseManager(address=(ip, port), authkey=b'abracadabra')
        connect(manager)
        self._in_queue = manager.get_queue(in_queue)
        self._out_queues = {k: manager.get_queue(k) for k in out_queues}

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

            if not blocking:
                while not self._out_queues[dest].empty():
                    try:
                        self._out_queues[dest].get(block=False)
                    except Empty:
                        break

            self._out_queues[dest].put(data[dest], block=blocking)

    def startup(self):
        pass

    def shutdown(self):
        pass

    @abstractmethod
    def loop(self, data: dict) -> dict:
        pass

    @_exception_handler
    @logger.catch(reraise=True)
    def run(self) -> None:
        self._startup()

        # This is to wait for the first message even in non-blocking mode
        data = self._recv()
        while True:
            data = self.loop(data)
            self._send_all(data, self.blocking)

            data = self._recv()

