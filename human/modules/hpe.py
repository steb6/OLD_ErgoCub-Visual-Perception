import copy
import pickle
from human.utils.misc import postprocess_yolo_output, homography, is_within_fov
import einops
import numpy as np
from utils.runner import Runner
import cv2


class HumanPoseEstimator:
    def __init__(self, model_config, cam_config):

        self.yolo_thresh = model_config.yolo_thresh
        self.nms_thresh = model_config.nms_thresh
        self.num_aug = model_config.num_aug

        # Intrinsics and K matrix of RealSense
        self.K = np.zeros((3, 3), np.float32)
        self.K[0][0] = cam_config.fx
        self.K[0][2] = cam_config.ppx
        self.K[1][1] = cam_config.fy
        self.K[1][2] = cam_config.ppy
        self.K[2][2] = 1

        # Load conversions
        self.skeleton = model_config.skeleton
        self.expand_joints = np.load(model_config.expand_joints_path)
        with open(model_config.skeleton_types_path, "rb") as input_file:
            self.skeleton_types = pickle.load(input_file)

        # Load modules
        self.yolo = Runner(model_config.yolo_engine_path)
        self.image_transformation = Runner(model_config.image_transformation_path)
        self.bbone = Runner(model_config.bbone_engine_path)
        self.heads = Runner(model_config.heads_engine_path)

    def estimate(self, frame):

        # Preprocess for yolo
        square_img = cv2.resize(frame, (256, 256), fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
        yolo_in = copy.deepcopy(square_img)
        yolo_in = cv2.cvtColor(yolo_in, cv2.COLOR_BGR2RGB)
        yolo_in = np.transpose(yolo_in, (2, 0, 1)).astype(np.float32)
        yolo_in = np.expand_dims(yolo_in, axis=0)
        yolo_in = yolo_in / 255.0

        # Yolo
        outputs = self.yolo(yolo_in)
        boxes, confidences = outputs[0].reshape(1, 4032, 1, 4), outputs[1].reshape(1, 4032, 80)
        bboxes_batch = postprocess_yolo_output(boxes, confidences, self.yolo_thresh, self.nms_thresh)

        # Get only the bounding box with the human with highest probability
        box = bboxes_batch[0]  # Remove batch dimension
        humans = []
        for e in box:  # For each object in the image
            if e[5] == 0:  # If it is a human
                humans.append(e)
        if len(humans) > 0:
            humans.sort(key=lambda x: x[4], reverse=True)  # Sort with decreasing probability
            human = humans[0]
        else:
            return None, None, None

        # Preprocess for BackBone
        x1 = int(human[0] * frame.shape[1]) if int(human[0] * frame.shape[1]) > 0 else 0
        y1 = int(human[1] * frame.shape[0]) if int(human[1] * frame.shape[0]) > 0 else 0
        x2 = int(human[2] * frame.shape[1]) if int(human[2] * frame.shape[1]) > 0 else 0
        y2 = int(human[3] * frame.shape[0]) if int(human[3] * frame.shape[0]) > 0 else 0
        new_K, homo_inv = homography(x1, x2, y1, y2, self.K, 256)

        # Apply homography
        H = self.K @ np.linalg.inv(new_K @ homo_inv)
        bbone_in = self.image_transformation([frame.astype(int), H.astype(np.float32)])

        bbone_in = bbone_in[0].reshape(1, 256, 256, 3)
        bbone_in_ = (bbone_in / 255.0).astype(np.float32)

        # Metro
        outputs = self.bbone(bbone_in_)  # HPE needs RGB
        logits = self.heads(outputs)

        # Get logits 3d
        logits = logits[0].reshape(1, 8, 8, 288)
        _, logits2d, logits3d = np.split(logits, [0, 32], axis=3)
        current_format = 'b h w (d j)'
        logits3d = einops.rearrange(logits3d, f'{current_format} -> b h w d j', j=32)  # 5, 8, 8, 9, 32

        # 3D Softmax
        heatmap_axes = (2, 1, 3)
        max_along_axis = logits3d.max(axis=heatmap_axes, keepdims=True)
        exponential = np.exp(logits3d - max_along_axis)
        denominator = np.sum(exponential, axis=heatmap_axes, keepdims=True)
        res = exponential / denominator

        # 3D Decode Heatmap
        result = []
        for ax in heatmap_axes:
            other_heatmap_axes = tuple(other_ax for other_ax in heatmap_axes if other_ax != ax)
            summed_over_other_heatmap_axes = np.sum(res, axis=other_heatmap_axes, keepdims=True)
            coords = np.linspace(0.0, 1.0, res.shape[ax])
            decoded = np.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
            result.append(np.squeeze(np.expand_dims(decoded, ax), axis=heatmap_axes))
        pred3d = np.stack(result, axis=-1)

        # 2D Softmax
        heatmap_axes = (2, 1)
        max_along_axis = logits2d.max(axis=heatmap_axes, keepdims=True)
        exponential = np.exp(logits2d - max_along_axis)
        denominator = np.sum(exponential, axis=heatmap_axes, keepdims=True)
        res = exponential / denominator

        # Decode heatmap
        result = []
        for ax in heatmap_axes:
            other_heatmap_axes = tuple(other_ax for other_ax in heatmap_axes if other_ax != ax)
            summed_over_other_heatmap_axes = np.sum(res, axis=other_heatmap_axes, keepdims=True)
            coords = np.linspace(0.0, 1.0, res.shape[ax])
            decoded = np.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
            result.append(np.squeeze(np.expand_dims(decoded, ax), axis=heatmap_axes))
        pred2d = np.stack(result, axis=-1) * 255

        # Get absolute position (if desired)
        is_predicted_to_be_in_fov = is_within_fov(pred2d)

        # If less than 1/4 of the joints is visible, then the resulting pose will be weird
        if is_predicted_to_be_in_fov.sum() < is_predicted_to_be_in_fov.size / 8:
            return None, None, None

        # Go back in original space (without augmentation and homography)
        pred3d = pred3d @ homo_inv

        # Get correct skeleton
        pred3d = (pred3d.swapaxes(1, 2) @ self.expand_joints).swapaxes(1, 2)
        if self.skeleton is not None:
            pred3d = pred3d[:, self.skeleton_types[self.skeleton]['indices']]
            edges = self.skeleton_types[self.skeleton]['edges']
        else:
            edges = None

        pred3d = pred3d[0]  # Remove batch dimension

        return pred3d, edges, (x1, x2, y1, y2)


if __name__ == "__main__":
    from human.utils.params import MetrabsTRTConfig, RealSenseIntrinsics
    from tqdm import tqdm
    from human.utils.matplotlib_visualizer import MPLPosePrinter
    from utils.input import RealSense

    vis = MPLPosePrinter()

    h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())

    cap = RealSense()
    # cap = cv2.VideoCapture(2)
    # cap = cv2.VideoCapture('assets/test_gaze_no_mask.mp4')

    for _ in tqdm(range(10000)):
    # while True:
        ret, img = cap.read()
        # cv2.imshow("img", img)
        # img = img[:, 240:-240, :]
        # img = cv2.resize(img, (640, 480))
        p, e, _ = h.estimate(img)
        if p is not None:
            p = p - p[0]
            vis.clear()
            vis.print_pose(p, e)
            vis.sleep(0.001)
