from multiprocessing.managers import BaseManager

import cv2

if __name__ == '__main__':
    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    q = manager.get_queue('grasping_sink')

    while True:
        data = q.get()
        cv2.imshow('', data['img'])
        cv2.waitKey(1)