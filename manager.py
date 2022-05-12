from multiprocessing.managers import BaseManager
from collections import defaultdict
from queue import Queue


queues = defaultdict(lambda: Queue(1))


def get_queue(name):
    return queues[name]


if __name__ == '__main__':

    BaseManager.register('get_queue', callable=get_queue)

    m = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    s = m.get_server()
    s.serve_forever()
