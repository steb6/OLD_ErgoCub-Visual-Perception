from multiprocessing import Queue
from multiprocessing.managers import BaseManager
from utils.input import RealSense


def main():
    # BaseManager.register('grasping')
    BaseManager.register('human')
    m = BaseManager(address=('localhost', 50000), authkey=b'qwerty')

    m.connect()
    # grasping: Queue = m.grasping()
    human: Queue = m.human()
    camera = RealSense()

    while True:
        _, img = camera.read()
        # grasping.put({'rgb': img, 'depth': None})
        human.put({'rgb': img})


if __name__ == '__main__':
    main()
