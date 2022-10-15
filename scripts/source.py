import copy
import multiprocessing
import time
from multiprocessing import Queue
from queue import Empty
from multiprocessing.managers import BaseManager, RemoteError
from typing import Dict, Union

import mediapipe.calculators.image.bilateral_filter_calculator_pb2
import cv2
import numpy as np
import pyrealsense2 as rs
from utils.input import RealSense
import sys
from loguru import logger

from utils.logging import get_logger
logger = get_logger(True)

@logger.catch(reraise=True)
def main():

    processes: Dict[str, Union[Queue, None]] = {'grasping': None}

    logger.info('Connecting to connection manager...')

    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    logger.success('Connected to connection manager')

    for proc in processes:
        processes[proc] = manager.get_queue(proc)

    file = 'assets/test_640.bag'
    camera = RealSense(color_format=rs.format.rgb8, fps=30, from_file=file)
    logger.info('Streaming to the connected processes...')

    # fps1 = 0
    # fps2 = 0
    i = 0
    debug = True
    while True:
        try:

            while True:
                start = time.perf_counter()
                # if i==0:  # TODO REMOVE DEBUG
                rgb, depth = camera.read()

                # fps1 += 1 / (time.perf_counter() - start)
                # print('read: ', fps1 / i)

                for queue in processes.values():
                    send(queue, {'rgb': copy.deepcopy(rgb), 'depth': copy.deepcopy(depth), 'debug': debug})

                # fps2 += 1 / (time.perf_counter() - start)
                # print('read + send', fps2/i)
                # i += 1
                cv2.imshow('Input', np.zeros([100, 100, 3], dtype=np.uint8))
                k = cv2.waitKey(1)
                if k == ord('d'):
                    debug = not debug
                    logger.info(f'Debugging {"on" if debug else "off"}')
                i += 1
        except RuntimeError as e:
            i = 0
            logger.error("Realsense: frame didn't arrive")
            # raise e
            # ctx = rs.context()
            # devices = ctx.query_devices()
            # for dev in devices:
            #     dev.hardware_reset()
            camera = RealSense(color_format=rs.format.rgb8, fps=30, from_file=file)




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
    # if not queue.empty():
    #     try:
    #         queue.get(block=False)
    #     except Empty:
    #         pass
    queue.put(data)


if __name__ == '__main__':
    main()
