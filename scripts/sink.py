from multiprocessing.managers import BaseManager

import cv2

from utils.logging import get_logger

logger = get_logger(True)

if __name__ == '__main__':
    logger.info('Connecting to connection manager...')

    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    logger.success('Connected to connection manager')

    q = manager.get_queue('grasping_sink')

    logger.info('Reading pipeline output')
    while True:
        data = q.get()
        cv2.imshow('', data['img'])
        cv2.waitKey(1)