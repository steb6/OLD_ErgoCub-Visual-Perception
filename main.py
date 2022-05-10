from multiprocessing import Queue
from multiprocessing.managers import BaseManager
from utils.input import RealSense
import time
import tqdm


def main():
    input("Press to continue")

    BaseManager.register('grasping')
    BaseManager.register('human')
    m = BaseManager(address=('localhost', 50000), authkey=b'qwerty')

    m.connect()
    grasping: Queue = m.grasping()
    human: Queue = m.human()
    camera = RealSense()

    while True:
        img, depth = camera.read()
        grasping.put({'rgb': img, 'depth': depth})
        human.put({'rgb': img})


if __name__ == '__main__':
    main()
