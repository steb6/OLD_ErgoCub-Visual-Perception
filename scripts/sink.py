import cv2
import numpy as np
from loguru import logger
from grasping.utils.misc import draw_mask
from gui.misc import project_pc, project_hands
from utils.concurrency import Node
from utils.logging import setup_logger
from configs.sink_config import Logging, Network

setup_logger(level=Logging.level)


@logger.catch(reraise=True)
class Sink(Node):
    def __init__(self):
        super().__init__(**Network.to_dict())

    def startup(self):
        logger.info('Starting up...')
        cv2.imshow('Ergocub-Visual-Perception', np.random.rand(480, 640, 3))
        logger.success('Start up complete.')

    def loop(self, data: dict) -> dict:
        if 'img' in data.keys():
            img = data['img']
        elif 'rgb' in data.keys():
            img = data['rgb']
        else:
            img = np.zeros([480, 640, 3], dtype=np.uint8)

        # GRASPING #####################################################################################################
        if 'mask' in data:
            img = draw_mask(img, data['mask'])

        if 'center' in data:
            img = cv2.circle(img, project_pc(data['center'])[0], 5, (0, 255, 0)).astype(np.uint8)

        if 'hands' in data:
            img = project_hands(img, data['hands']['right'], data['hands']['left'])

        # HUMAN ########################################################################################################
        if 'fps' in data:
            img = cv2.putText(img, f'FPS: {int(data["fps"])}', (10, 20), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1,
                              cv2.LINE_AA)

        if 'distance' in data:
            img = cv2.putText(img, f'DIST: {int(data["distance"])}', (200, 20), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1,
                              cv2.LINE_AA)

        if 'focus' in data:
            img = cv2.putText(img, "FOCUS" if data["focus"] else "NOT FOCUS", (400, 20), cv2.FONT_ITALIC, 0.7,
                              (0, 255, 0) if data["focus"] else (0, 0, 255), 1, cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('Ergocub-Visual-Perception', img)
        cv2.setWindowProperty('Ergocub-Visual-Perception', cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        return {}


if __name__ == '__main__':
    source = Sink()
    source.run()
