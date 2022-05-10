from multiprocessing.managers import BaseManager
from collections import defaultdict
from queue import Queue

queues = defaultdict(Queue)
class QueueManager(BaseManager): pass
QueueManager.register('get_queue', callable=lambda uuid:queues[uuid])
m = QueueManager(address=('localhost', 50000), authkey=b'abracadabra')
s = m.get_server()
s.serve_forever()