import copy
import multiprocessing
import time
from multiprocessing import Queue
from queue import Empty
from multiprocessing.managers import BaseManager, RemoteError
from typing import Dict, Union
import pyrealsense2 as rs
from utils.input import RealSense
import sys
from loguru import logger


logger.remove()
logger.add(sys.stdout,
           format="<fg #b28774>{time:YYYY-MM-DD HH:mm:ss:SSS ZZ}</> <yellow>|</>"
                  " <lvl>{level: <8}</> "
                  "<yellow>|</> <blue>{process.name: ^12}</> <yellow>-</> <lvl>{message}</>",
           diagnose=True)

logger.level('INFO', color='<fg #fef5ed>')
logger.level('SUCCESS', color='<fg #79d70f>')
logger.level('WARNING', color='<fg #fd811e>')
logger.level('ERROR', color='<fg #ed254e>')


@logger.catch
def main():
    set_name('Source')

    processes: Dict[str, Union[Queue, None]] = {'grasping': None, 'human': None}

    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    for proc in processes:
        processes[proc] = manager.get_queue(proc)

    camera = RealSense(color_format=rs.format.rgb8, fps=30)
    logger.info('Streaming to the connected processes...')

    fps = 0
    i = 0
    while True:
        try:

            while True:
                start = time.perf_counter()
                rgb, depth = camera.read()

                for queue in processes.values():
                    send(queue, {'rgb': copy.deepcopy(rgb), 'depth': copy.deepcopy(depth)})

                fps += 1 / (time.perf_counter() - start)
                i += 1
                print('\r', fps/i, end='')
        except RuntimeError:
            logger.error("Realsense: frame didn't arrive")
            # ctx = rs.context()
            # devices = ctx.query_devices()
            # for dev in devices:
            #     dev.hardware_reset()
            camera = RealSense(color_format=rs.format.rgb8, fps=30)


def set_name(name):
    multiprocessing.current_process().name = name


def connect(manager):
    logger.info('Connecting to manager...')
    start = time.time()

    while True:
        try:
            manager.connect()
            break
        except ConnectionRefusedError as e:
            if time.time() - start > 120:
                logger.error('Connection refused.')
                raise e
            time.sleep(1)
    logger.success('Connected to manager.')


def register(manager, processes):
    for proc in list(processes):
        BaseManager.register(proc)
        try:
            processes[proc] = getattr(manager, proc)()
        except RemoteError:
            logger.warning(f"Couldn't connect to process '{proc}'")
            processes.pop(proc)

    logger.success(f'Connected to: {",".join(list(processes))}')
    return processes


def send(queue, data):
    if not queue.empty():
        try:
            queue.get(block=False)
        except Empty:
            pass
    queue.put(data)


if __name__ == '__main__':
    main()
