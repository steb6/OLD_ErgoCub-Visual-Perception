import queue
import sys
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue

from loguru import logger

from utils.multiprocessing import DataManager, Signals

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
            pm = DataManager(list(args[0]._networks.values()))
            pm.write(signal=Signals.STOP, important=True)
            logger.info('Caught an exception')
            for net in args[0]._networks.values():
                net.broadcast(pm)
            logger.exception('Exception Raised')
            exit(1)

    return wrapper


class Node(Process, ABC):

    def __init__(self, name, blocking=True):
        super(Process, self).__init__()

        self.blocking = blocking

        self._in_queue = Queue(1)
        self._networks = {}

        self.name = name

    def _startup(self):
        logger.info('Starting up...')

        self.startup()

        logger.info('Waiting for source startup...')

        data = self._recv()
        if data.ready(self.name):
            self._send_all(data)
        else:
            raise RuntimeError('Cannot startup if the source node is not ready')

        logger.info('Start up complete.')

    def _shutdown(self, data):
        logger.info('Shutting down...')

        self.shutdown()

        if data.stop(self.name):
            self._send_all(data)
        else:
            raise RuntimeError('Last message should contain the stop signal')
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

        make_copy = True
        if len(list(self._networks)) == 1:
            make_copy = False

        for net in list(self._networks):

            out_queues = self._networks[net].connections[self.name]
            names = self._networks[net].map[self.name]

            data = data.dispatch(net, make_copy)

            for out, name in zip(out_queues, names):
                try:
                    old_data = out.get_nowait()
                    if old_data.important():
                        out.put(old_data)

                except queue.Empty:
                    pass
                try:
                    if data.important():
                        out.put(data, timeout=10)
                    else:
                        out.put_nowait(data)
                except queue.Full:
                    pass

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
            if data.stop(self.name):
                break
            data = self.loop(data)
            self._send_all(data)

            if self.blocking:
                data = self._recv()
            else:
                res = self._recv_nowait()
                if res is not None:
                    data = res

        self._shutdown(data)


pm = {
    'network1': {
        'node1': {'from1': 'msg', 'from2': 'msg'},  # messages for node 1
        'node2': {'from1': 'msg', 'from2': 'msg'},
    },
    'network2': {
        'node1': {'from1': 'msg', 'from2': 'msg'},
        'node2': {'from1': 'msg', 'from2': 'msg'},
    }
}
