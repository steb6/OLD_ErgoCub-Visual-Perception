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
        self.box_center = np.array([0, 0, 0])

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

        # Center
        pose3d_root = pose3d_abs - pose3d_abs[0, :] if pose3d_abs is not None else None

        # Get pose
        poses = self.pr.inference(pose3d_root)
        pose = None
        if len(poses) > 0:
            pose = list(poses.keys())[list(poses.values()).index(max(poses.values()))]

        # Get box position  # TODO CHANGE BOX-HUMAN CHECK
        box_position = self.grasping_queue.get()
        # box_position = np.array([1, 1, 1])
        box_distance = -1
        if np.any(box_position):
            box_distance = np.linalg.norm(np.array([0, 0, 0]) - box_position)

        # Select manually correct action
        actions = []
        poi = None
        if pose is not None:
            if pose == "stand":
                actions.append(f"stand: {poses[pose]}")
            if pose == "safe":
                actions.append(f"safe: {poses[pose]}")
            if pose == "unsafe":
                actions.append(f"unsafe: {poses[pose]}")
            if pose == "hello" and focus:
                actions.append(f"hello: {poses[pose]}")
            if pose == "wait" and focus and box_distance == -1:
                actions.append(f"give: {poses[pose]}")
        if box_distance != -1 and box_distance < 0.7:  # Less than 70 cm
            actions.append("get")
        print(actions)

        # Get box center
        elements = {"img": img,
                    "pose": pose3d_root,
                    "edges": edges,
                    "focus": focus,
                    "actions": actions,
                    "distance": d,  # TODO fix
                    "human_bbox": human_bbox,
                    "face_bbox": face_bbox,
                    "box_center": box_position
                    }

        # # Compute fps
        # end = time.time()
        # self.fps_s.append(1. / (end - start) if (end-start) != 0 else 0)
        # fps_s = self.fps_s[-10:]
        # fps = sum(fps_s) / len(fps_s)
        # print(fps)

        return elements

    def shutdown(self):
        pass
