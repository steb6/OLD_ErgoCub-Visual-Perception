from multiprocessing.managers import BaseManager

import cv2
import numpy as np
from loguru import logger

from grasping.utils.misc import draw_mask
from gui.misc import project_pc, project_hands
from utils.logging import setup_logger

from configs.sink_config import Logging

setup_logger(level=Logging.level)

if __name__ == '__main__':
    logger.info('Connecting to connection manager...')

    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    logger.success('Connected to connection manager')

    q = manager.get_queue('grasping_sink')

    logger.info('Reading pipeline output')

    # cv2.namedWindow('grasping_output', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('grasping_output', cv2.WND_PROP_TOPMOST, 1)
    i = 0
    while True:
        data = q.get()

        if 'rgb' in data:
            img = data['rgb']
        else:
            img = np.zeros([480, 640, 3], dtype=np.uint8)

        if 'mask' in data:
            img = draw_mask(img, data['mask'])

        if 'center' in data:
            img = cv2.circle(img, project_pc(data['center'])[0], 5, (0, 255, 0)).astype(np.uint8)

        if 'hands' in data:
            img = project_hands(img, data['hands']['right'], data['hands']['left'])

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if 'fps' in data:
            img = cv2.putText(img, f'FPS: {int(data["fps"])}', (10, 20), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1,
                              cv2.LINE_AA)

        cv2.imshow('grasping_output', img)
        cv2.waitKey(1)

        cv2.imwrite(f'out{i}.png', img)
        i += 1

        print()