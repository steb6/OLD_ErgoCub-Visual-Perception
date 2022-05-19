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

        self.grasping_queue = self.manager.get_queue('grasping_out')
        self.last_box_position = np.array([0, 0, 0])

    def loop(self, data):
        img = data['rgb']
        start = time.time()

        # Start independent modules
        self.hpe_in.put(img)
        self.focus_in.put(img)
        pose3d_abs, edges, human_bbox = self.hpe_out.get()
        focus_ret = self.focus_out.get()

        # Focus
        focus = False
        face_bbox = None
        if focus_ret is not None:
            focus, face_bbox = focus_ret

        # Compute distance
        d = None
        if pose3d_abs is not None:
            cam_pos = np.array([0, 0, 0])
            man_pose = np.array(pose3d_abs[0])
            d = np.sqrt(np.sum(np.square(cam_pos - man_pose)))

        # Get pose
        pose3d_root = pose3d_abs - pose3d_abs[0, :] if pose3d_abs is not None else None
        poses = self.pr.inference(pose3d_root)
        pose = None
        if len(poses) > 0:
            pose = list(poses.keys())[list(poses.values()).index(max(poses.values()))]

        # Get box position
        # Wait
        box_position = self.grasping_queue.get()
        # No Wait
        # box_position = self.grasping_queue.get() if not self.grasping_queue.empty() else self.last_box_position
        self.last_box_position = box_position
        box_distance = -1
        if np.any(box_position):
            box_distance = np.linalg.norm(np.array([0, 0, 0]) - box_position)

        # Select manually correct action
        action = None
        if pose is not None:
            if poses[pose] > 0.7:  # Filter uncertainty
                if pose == "stand":
                    action = "stand: {:.2f}".format(poses[pose])
                if pose == "safe" and box_distance != -1:
                    action = "safe: {:.2f}".format(poses[pose])
                if pose == "unsafe" and box_distance != -1:
                    action = "unsafe: {:.2f}".format(poses[pose])
                if pose == "hello" and focus:
                    action = "hello: {:.2f}".format(poses[pose])
                if pose == "wait" and focus and box_distance == -1:
                    action = "give: {:.2f}".format(poses[pose])
        if box_distance != -1 and box_distance < 0.7:  # Less than 70 cm
            action = "get (dist: {:.2f}".format(box_distance)
        print(action)

        # Get box center
        elements = {"img": img,
                    "pose": pose3d_root,
                    "edges": edges,
                    "focus": focus,
                    "action": action,
                    "distance": d,  # TODO fix
                    "human_bbox": human_bbox,
                    "face_bbox": face_bbox,
                    "box_center": box_position
                    }

        # # Compute fps
        end = time.time()
        self.fps_s.append(1. / (end - start) if (end-start) != 0 else 0)
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)
        print(fps)

        return elements

    def shutdown(self):
        pass
