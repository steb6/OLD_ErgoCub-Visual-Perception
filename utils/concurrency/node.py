import queue
import sys
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager

from loguru import logger
from queue import Empty

# from utils.multiprocessing import DataManager, Signals

logger.remove()
logger.add(sys.stdout,
           format="<blue>{time:YYYY-MM-DD HH:mm:ss:SSS ZZ}</> |"
                  " {level: <8} "
                  "| <yellow>{process.name: ^12}</> - {message}",
           diagnose=True)

def _exception_handler(function):
    def wrapper(*args):
        try:
            function(*args)
        except:
            # pm = DataManager(list(args[0]._networks.values()))
            # pm.write(signal=Signals.STOP, important=True)

            # for net in args[0]._networks.values():
                # net.broadcast(pm)
            logger.exception('Exception Raised')
            exit(1)

    return wrapper


class Node(Process, ABC):

    def __init__(self, name, blocking=True):
        super(Process, self).__init__()

        self.name = name
        self.blocking = blocking

        BaseManager.register('get_queue')
        manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
        manager.connect()
        self._in_queue = manager.get_queue(self.name)

        self._out_queue = manager.get_queue(f'vis_in_{self.name}')



    def _startup(self):
        logger.info('Starting up...')

        self.startup()

        logger.info('Waiting for source startup...')

        data = self._recv()

        # self._send_all(data)

        logger.info('Start up complete.')

    def _shutdown(self, data):
        logger.info('Shutting down...')

        self.shutdown()

        # self._send_all(data)

        logger.info('Shut down complete.')

    def _recv(self):

        data = self._in_queue.get()

        return data

    def _recv_nowait(self):
        if not self._in_queue.empty():
            return self._recv()
        else:
            return None

    def _send_all(self, data):
        while not self._out_queue.empty():
            try:
                self._out_queue.get(block=False)
            except Empty:
                break

        self._out_queue.put(data)

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
            self._send_all(data)

            if self.blocking:
                data = self._recv()
            else:
                res = self._recv_nowait()
                if res is not None:
                    data = res

        self._shutdown(data)

    def connect(self, name, queue):
        self._out_queues[name] = queue

    def get_input(self):
        return self._in_queue
