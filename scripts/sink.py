import cv2
import numpy as np
from loguru import logger
from grasping.utils.misc import draw_mask, project_pc, project_hands

from utils.concurrency import Node
from utils.logging import setup_logger
from configs.sink_config import Logging, Network

setup_logger(**Logging.Logger.Params.to_dict())


@logger.catch(reraise=True)
class Sink(Node):
    def __init__(self):
        self.img = np.zeros([480, 640, 3], dtype=np.uint8)
        self.mask = None
        self.center = None
        self.hands = None
        self.fps = None
        self.distance = None
        self.focus = None
        self.pose = None
        self.bbox = None
        self.face_bbox = None
        self.actions = None
        self.edges = None
        super().__init__(**Network.to_dict())

    # def startup(self):
    #     logo = cv2.imread('assets/logo_transparent.png')
    #     logo = cv2.resize(logo, (640, 480))
    #     cv2.imshow('Ergocub-Visual-Perception', logo)
    #     cv2.waitKey(1)

    def loop(self, data: dict) -> dict:
        if 'img' in data.keys():
            self.img = data['img']
        img = self.img

        # GRASPING #####################################################################################################
        if 'mask' in data.keys():
            self.mask = data['mask']
        img = draw_mask(img, self.mask)

        if 'center' in data:
            self.center = data['center']
        if self.center is not None:
            img = cv2.circle(img, project_pc(self.center)[0], 5, (0, 255, 0)).astype(np.uint8)

        if 'hands' in data:
            self.hands = data['hands']
        if self.hands is not None:
            img = project_hands(img, self.hands['right'], self.hands['left'])

        # HUMAN ########################################################################################################
        if 'fps' in data:
            self.fps = data['fps']
        if self.fps is not None:
            img = cv2.putText(img, f'FPS: {int(self.fps)}', (10, 20), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1,
                              cv2.LINE_AA)

        if 'distance' in data:
            self.distance = data['distance']
        if self.distance is not None:
            img = cv2.putText(img, f'DIST: {int(self.distance)}', (200, 20), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1,
                              cv2.LINE_AA)

        if 'focus' in data:
            self.focus = data['focus']
        if self.focus is not None:
            img = cv2.putText(img, "FOCUS" if self.focus else "NOT FOCUS", (400, 20), cv2.FONT_ITALIC, 0.7,
                              (0, 255, 0) if self.focus else (0, 0, 255), 1, cv2.LINE_AA)

        if 'pose' in data:
            self.pose = data["pose"]
            self.edges = data["edges"]
        if self.pose is not None:
            img = cv2.rectangle(img, (0, 0), (100, 100), (255, 255, 255), cv2.FILLED)
            for edge in self.edges:
                p0 = [int((p*50)+50) for p in self.pose[edge[0]][:2]]
                p1 = [int((p*50)+50) for p in self.pose[edge[1]][:2]]
                img = cv2.line(img, p0, p1, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        if 'bbox' in data:
            self.bbox = data["bbox"]
        if self.bbox is not None:
            x1, x2, y1, y2 = self.bbox
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if 'focus' in data.keys():
            self.focus = data["focus"]
        if self.focus is not None:
            focus = self.focus
        else:
            focus = False

        if 'face_bbox' in data.keys():
            self.face_bbox = data["face_bbox"]
        if self.face_bbox is not None:
            x1, y1, x2, y2 = self.face_bbox
            color = (255, 0, 0) if not focus else (0, 255, 0)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        if 'actions' in data.keys():
            self.actions = data["actions"]
        if 'is_true' in data.keys():
            self.is_true = data["is_true"]
        if self.actions is not None:
            if len(self.actions) > 1:
                best = max(self.actions, key=self.actions.get)
                if self.is_true > 0.75:
                    img = cv2.putText(img, best, (10, 100), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1, cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('Ergocub-Visual-Perception', img)
        cv2.setWindowProperty('Ergocub-Visual-Perception', cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        return {}


if __name__ == '__main__':
    source = Sink()
    source.run()
