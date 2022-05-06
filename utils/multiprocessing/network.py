import queue
from multiprocessing import Queue

from loguru import logger

from utils.multiprocessing import Signals, DataManager


class Network:
    def __init__(self, name='next', source=None, map: dict = None, sink=None, start=True):
        self.name = name
        self.source = {'name': source.name, 'queue': source._in_queue}
        self.sink = {'name': sink.name, 'queue': sink._in_queue}

        self.inputs = {node.name: node._in_queue for node in map}

        for k, v in map.items():
            if not isinstance(v, list):
                map[k] = [v]

        self.map = {inps.name: [out.name for out in outs] for inps, outs in map.items()}

        self.connections = self.build(map)

        if start:
            for node in list(map):  # TODO tapullo risolvere meglio
                node.start()

            data = DataManager(self)
            data.write(signal=Signals.READY)
            self.send(data)
            data = self.recv()

            if not data.ready(self.sink['name']):
                raise ValueError('Startup not successful')

    # TODO model different cases where this approach fails
    def build(self, map):
        connections = {}

        for src, dests in map.items():

            for dest in dests:
                if src.name not in connections:
                    connections[src.name] = []
                connections[src.name].append(self.connect(src, dest))

        return connections

    def connect(self, src, dest):
        if dest._in_queue is None:
            dest._in_queue = Queue(1)

        link = dest._in_queue

        src._networks[self.name] = self
        dest._networks[self.name] = self

        return link

    def send(self, data):
        self.source['queue'].put(data)

    def recv(self):
        data = self.sink['queue'].get()

        return data

    def broadcast(self, data):

        logger.info(f'Broadcasting to {list(self.inputs)}')
        for name, in_queue in self.inputs.items():

            try:
                old_data = in_queue.get_nowait()
                if old_data.important():
                    data = old_data

            except queue.Empty:
                pass

            in_queue.put(data)