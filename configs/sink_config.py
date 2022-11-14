from logging import INFO

from utils.confort import BaseConfig


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = True


class Network(BaseConfig):
    ip = 'localhost'
    port = 50000
    in_queue = 'sink'
    out_queues = []

