from multiprocessing import Queue
from multiprocessing.managers import BaseManager

import cv2


def main():
    BaseManager.register('grasping')
    m = BaseManager(address=('localhost', 50000), authkey=b'qwerty')

    m.connect()
    grasping: Queue = m.grasping()
    camera = cv2.VideoCapture(0)

    while True:
        _, img = camera.read()
        grasping.put({'rgb': img, 'depth': None})


if __name__ == '__main__':
    main()