import copy

from utils.multiprocessing import Signals


class DataManager:
    def __init__(self, networks):
        if not isinstance(networks, list):
            networks = [networks]

        self.networks = {net.name: list(net.connections) for net in networks}
        # TODO maybe make it more readeble
        self._messages = {net.name:
            {
                node: {} for node in
                set([x for inp, outs in net.map.items() for x in outs + [inp]])  # all nodes in the net
            }
            for net in networks}

        # self._debug = {worker.name: debug for worker in workers}
        # self._signals = {worker.name: signal for worker in workers}
        # self.workers = [worker.name for worker in workers]

    # TODO Add type hints
    def write(self, message: dict = None, frm='', to='all', through='next', signal=Signals.OK, debug=False,
              important=False):

        if through == 'all':
            through = list(self.networks)
        elif not isinstance(through, list):
            through = [through]

        for net in through:

            nodes = list(self._messages[net])

            if to == 'all':
                to = nodes

            if not isinstance(to, list):
                to = [to]

            for addr in to:
                if addr in list(self._messages[net]):
                    if message:
                        self._messages[net][addr][frm] = message
                    self._messages[net][addr]['_signal'] = signal
                    self._messages[net][addr]['_debug'] = debug
                    self._messages[net][addr]['_important'] = important
                    pass
                else:
                    raise ValueError('Unknown recipient')

    def update(self, message: dict = None, frm='', to='all', through='next', signal=None, debug=None,
              important=None):

        if through == 'all':
            through = list(self.networks)
        elif not isinstance(through, list):
            through = [through]

        for net in through:

            nodes = list(self._messages[net])

            if to == 'all':
                to = nodes

            if not isinstance(to, list):
                to = [to]

            # logger.info(list(self._messages[net]))
            # logger.info(to)
            # logger.info(message)
            for addr in to:
                if addr in list(self._messages[net]):
                    if message:

                        # if addr == 'debug':
                            # logger.info(net, addr, frm)
                            # logger.info(self._messages)
                        self._messages[net][addr][frm] = message
                        # if addr == 'debug':
                        #     logger.info(self._messages)

                    if signal:
                        self._messages[net][addr]['_signal'] = signal
                    if debug:
                        self._messages[net][addr]['_debug'] = debug
                    if important:
                        self._messages[net][addr]['_important'] = important
                    pass
                else:
                    raise ValueError('Unknown recipient')

    # A message is cloned when it goes through multiple networks
    #   this is to assure that they are not accessed concurrently.
    # TODO a network can have multiple output and that is not being managed. How should this be managed?
    def dispatch(self, network, make_copy=False):
        if make_copy:
            pm = copy.deepcopy(self)
        else:
            pm = self

        pm._messages = {network: pm._messages[network]}
        return pm

    def read(self, net, addr):
        # TODO add net='all'
        return self._messages[net][addr]

    def remove(self, net, addr):
        self._messages[net].pop(addr)

    def debug(self, node) -> bool:
        for net in self.networks:
            if self._messages[net][node]['_debug']:
                return True
        return False

    def important(self):
        for net in self.networks:
            for node in list(self._messages[net]):
                if self._messages[net][node]['_important']:
                    return True
        return False

    def stop(self, node) -> bool:
        for net in self.networks:
            if self._messages[net][node]['_signal'] == Signals.STOP:
                return True
        return False

    def ready(self, node) -> bool:
        for net in self.networks:
            if self._messages[net][node]['_signal'] == Signals.READY:
                return True
        return False
