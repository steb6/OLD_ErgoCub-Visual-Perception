import cv2
from grasping.modules.utils.misc import draw_mask
from gui.misc import project_pc, project_hands
from human.modules.focus import FocusDetector
from human.modules.pr import PoseRecognizer
from human.modules.hpe import HumanPoseEstimator
from human.utils.params import FocusConfig, MetrabsTRTConfig, RealSenseIntrinsics, TRXConfig
from utils.concurrency import Node
import time
import numpy as np
from multiprocessing import Queue, Process


def run_module(module, configurations, input_queue, output_queue):
    x = module(*configurations)
    while True:
        inp = input_queue.get()
        y = x.estimate(inp)
        output_queue.put(y)


class Human(Node):
    def __init__(self):
        super().__init__(name='human')
        self.focus_in = None
        self.focus_out = None
        self.focus_proc = None
        self.hpe_in = None
        self.hpe_out = None
        self.hpe_proc = None
        self.pr = None
        self.window_size = None
        self.fps_s = None
        self.last_poses = None
        self.grasping_queue = None
        self.box_center = None
        self.last_box_position = None
        cv2.namedWindow("Ergocub-Visual-Perception")  # Create a named window
        cv2.moveWindow("Ergocub-Visual-Perception", 40, 30)  # Move it to (40,30)

    def startup(self):
        # Load modules
        self.focus_in = Queue(1)
        self.focus_out = Queue(1)
        self.focus_proc = Process(target=run_module, args=(FocusDetector,
                                                           (FocusConfig(),),
                                                           self.focus_in, self.focus_out))
        self.focus_proc.start()

        self.hpe_in = Queue(1)
        self.hpe_out = Queue(1)
        self.hpe_proc = Process(target=run_module, args=(HumanPoseEstimator,
                                                         (MetrabsTRTConfig(), RealSenseIntrinsics()),
                                                         self.hpe_in, self.hpe_out))
        self.hpe_proc.start()

        self.pr = PoseRecognizer(TRXConfig())

        self.fps_s = []
        self.last_poses = []

        self.grasping_queue = self.manager.get_queue('grasping_human')
        self.last_box_position = np.array([0, 0, 0])

        self._out_queues['visualizer'] = self.manager.get_queue('human_visualizer')

    def loop(self, data):
        img = data['rgb']
        start = time.time()

        # Start independent modules
        self.hpe_in.put(img)
        self.focus_in.put(img)
        pose3d_abs, pose2d, edges, human_bbox = self.hpe_out.get()
        focus_ret = self.focus_out.get()

        # Focus
        focus = None
        face = None
        if focus_ret is not None:
            focus, face = focus_ret

        # Compute distance
        d = None
        if pose3d_abs is not None:
            d = np.linalg.norm(pose3d_abs[0] * 2.2)  # Metrabs denormalization

        # Get pose
        pose3d_root = pose3d_abs - pose3d_abs[0, :] if pose3d_abs is not None else None
        poses = self.pr.inference(pose3d_root)
        pose = None
        if len(poses) > 0:
            pose = list(poses.keys())[list(poses.values()).index(max(poses.values()))]

        # Get box position
        # Wait
        grasping_data = self.grasping_queue.get()
        box_position = grasping_data['center']
        # No Wait
        # box_position = self.grasping_queue.get() if not self.grasping_queue.empty() else self.last_box_position
        # self.last_box_position = box_position
        box_distance = -1
        if box_position is not None:
            box_distance = np.linalg.norm(box_position)

        # Select manually correct action
        action = None
        if pose is not None:
            if poses[pose] > 0.6:  # Filter uncertainty
                if pose == "stand":
                    action = "stand: {:.2f}".format(poses[pose])
                if pose == "safe" and box_distance != -1:
                    action = "safe: {:.2f}".format(poses[pose])
                if pose == "unsafe" and box_distance != -1:
                    action = "unsafe: {:.2f}".format(poses[pose])
                if pose == "hello" and focus is not None and focus:
                    action = "hello: {:.2f}".format(poses[pose])
                if pose == "wait" and focus is not None and focus and box_distance == -1:
                    action = "give: {:.2f}".format(poses[pose])
        if box_distance != -1 and box_distance < 0.7:  # Less than 70 cm
            action = "get (dist: {:.2f})".format(box_distance)

        # Get box center
        elements = {}
        if 'debug' in data and data['debug']:
            elements['visualizer'] = {"img": img,
                                      "pose": pose3d_abs * 2.2 if pose3d_abs is not None else None,
                                      "edges": edges,
                                      "focus": focus,
                                      "action": action,
                                      "distance": d,
                                      "human_bbox": human_bbox,
                                      "face": face,
                                      "box_center": box_position,
                                      "pose2d": pose2d,
                                      "grasping": grasping_data,
                                      }

        # # Compute fps
        end = time.time()
        self.fps_s.append(1. / (end - start) if (end - start) != 0 else 0)
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)

        # Light visualizer
        if human_bbox is not None:
            x1, x2, y1, y2 = human_bbox
            img = cv2.rectangle(img,
                                (x1, y1), (x2, y2), (0, 0, 255), 1).astype(np.uint8)
        if face is not None:
            x1, y1, x2, y2 = face.bbox.reshape(-1)
            img = cv2.rectangle(img,
                                (x1, y1), (x2, y2), (255, 0, 0), 1).astype(np.uint8)
        if box_position is not None:
            img = cv2.circle(img, project_pc(box_position)[0], 5, (0, 255, 0)).astype(np.uint8)
        if pose2d is not None:
            for edge in edges:
                c1 = 0 < pose2d[edge[0]][0] < 640 and 0 < pose2d[edge[0]][1] < 480
                c2 = 0 < pose2d[edge[1]][0] < 640 and 0 < pose2d[edge[1]][1] < 480
                if c1 and c2:
                    img = cv2.line(img, pose2d[edge[0]], pose2d[edge[1]], (255, 0, 255), 1, cv2.LINE_AA)

        if grasping_data['mask'] is not None:
            img = draw_mask(img, grasping_data['mask'])

        if grasping_data['hands'] is not None:
            hands = grasping_data['hands']
            img = project_hands(img, hands['right'], hands['left'])

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.putText(img, "FPS: {}".format(int(fps)), (10, 20), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1,
                          cv2.LINE_AA)
        img = cv2.putText(img, "{}".format(action) if action is not None else "",
                          (215, 20), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1,
                          cv2.LINE_AA)
        img = cv2.putText(img, "Focus: {}".format(focus) if focus is not None else "",
                          (420, 20), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1,
                          cv2.LINE_AA)
        cv2.imshow("Ergocub-Visual-Perception", img)
        cv2.waitKey(1)

        return elements

    def shutdown(self):
        pass
