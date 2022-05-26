import copy
import pickle
from human.utils.misc import postprocess_yolo_output, homography, is_within_fov, reconstruct_absolute, \
    is_pose_consistent_with_box, get_augmentations
import einops
import numpy as np

from human.utils.tracking import Sort
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

        # SMPL+HEAD_30 mirror mapping
        self.joint_mirror_map = np.array([[1, 2],
                                          [4, 5],
                                          [7, 8],
                                          [10, 11],
                                          [13, 14],
                                          [16, 17],
                                          [18, 19],
                                          [20, 21],
                                          [22, 23],
                                          [25, 26],
                                          [28, 29]])

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

        # Load tracking
        self.tracking = Sort()

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
            return None, None, None, None

        # Preprocess for BackBone
        x1 = int(human[0] * frame.shape[1]) if int(human[0] * frame.shape[1]) > 0 else 0
        y1 = int(human[1] * frame.shape[0]) if int(human[1] * frame.shape[0]) > 0 else 0
        x2 = int(human[2] * frame.shape[1]) if int(human[2] * frame.shape[1]) > 0 else 0
        y2 = int(human[3] * frame.shape[0]) if int(human[3] * frame.shape[0]) > 0 else 0

        # Tracking
        res = self.tracking.update(np.array([x1, y1, x2, y2, human[4]])[None, ...])
        if len(res) > 0:
            x1, y1, x2, y2, _ = res[0].astype(int)
        new_K, homo_inv = homography(x1, x2, y1, y2, self.K, 256)

        # Test time augmentation TODO ADD GAMMA DECODING
        if self.num_aug > 0:
            aug_should_flip, aug_rotflipmat, aug_gammas, aug_scales = get_augmentations(self.num_aug)
            new_K = np.tile(new_K, (self.num_aug, 1, 1))
            for k in range(self.num_aug):
                new_K[k, :2, :2] *= aug_scales[k]
            homo_inv = aug_rotflipmat @ np.tile(homo_inv[0], (self.num_aug, 1, 1))

        # Apply homography
        H = self.K @ np.linalg.inv(new_K @ homo_inv)
        bbone_in = self.image_transformation([frame.astype(int), H.astype(np.float32)])

        bbone_in = bbone_in[0].reshape(self.num_aug, 256, 256, 3)  # [0] to get from list
        bbone_in_ = (bbone_in / 255.0).astype(np.float32)

        # Metro
        outputs = self.bbone(bbone_in_)  # HPE needs RGB
        logits = self.heads(outputs)

        # Get logits 3d
        logits = logits[0].reshape(self.num_aug, 8, 8, 288)
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
        # TODO START DEBUG is_within_fov value
        # for img, res, points in zip(bbone_in[2:3], is_predicted_to_be_in_fov[2:3], pred2d[2:3]):
        #     for flag, point in zip(res, points):
        #         img = cv2.circle(img, point.astype(int), 5, (255, 0, 0) if flag else (0, 0, 255))
        #     cv2.imshow("", img.astype(np.uint8))
        #     cv2.waitKey(1)
        # TODO END DEBUG
        if is_predicted_to_be_in_fov.sum() < is_predicted_to_be_in_fov.size / 4:
            return None, None, None, None
        # TODO TRY TO USE REAL INTRINSICS AND MULTIPLY WITH DIFFERENT VALUE

        # Move the skeleton into estimated absolute position if necessary
        pred3d = reconstruct_absolute(pred2d, pred3d, new_K, is_predicted_to_be_in_fov,
                                      weak_perspective=False)

        # Go back in original space (without augmentation and homography)
        pred3d = pred3d @ homo_inv

        # Get correct skeleton
        pred3d = (pred3d.swapaxes(1, 2) @ self.expand_joints).swapaxes(1, 2)
        if self.skeleton is not None:
            pred3d = pred3d[:, self.skeleton_types[self.skeleton]['indices']]
            edges = self.skeleton_types[self.skeleton]['edges']
        else:
            edges = None

        # Mirror predictions
        if self.num_aug > 0:
            for k in range(len(self.joint_mirror_map)):
                aux = pred3d[aug_should_flip, self.joint_mirror_map[k, 0]]
                pred3d[aug_should_flip, self.joint_mirror_map[k, 0]] = pred3d[aug_should_flip,
                                                                              self.joint_mirror_map[k, 1]]
                pred3d[aug_should_flip, self.joint_mirror_map[k, 1]] = aux

        # Aggregate results
        pred3d = pred3d.mean(axis=0)

        # Project skeleton over image
        pose2d, _ = cv2.projectPoints(pred3d * 2.2, np.array([0, 0, 0], dtype=np.float32)[None, ...],
                                      np.array([0, 0, 0], dtype=np.float32)[None, ...],
                                      self.K,
                                      np.array([[0.], [0.], [0.], [0.], [0.]]))
        pose2d = pose2d.astype(int)
        pose2d = pose2d[:, 0, :]

        # Check if results are consistent with bounding box  # TODO SHOULD FILTER AUGMENTATIONS
        if not is_pose_consistent_with_box(pose2d, (x1, y1, x2, y2)):
            return None, None, None, None

        return pred3d, pose2d, edges, (x1, x2, y1, y2)


if __name__ == "__main__":
    from human.utils.params import MetrabsTRTConfig, RealSenseIntrinsics
    from tqdm import tqdm
    from human.utils.matplotlib_visualizer import MPLPosePrinter
    from utils.input import RealSense

    vis = MPLPosePrinter()

    h = HumanPoseEstimator(MetrabsTRTConfig(), RealSenseIntrinsics())

    # cap = RealSense()
    cap = cv2.VideoCapture(2)
    # cap = cv2.VideoCapture('recording.mp4')

    for _ in tqdm(range(10000)):
    # while True:
    #     img, depth = cap.read()
        ret, img = cap.read()
        p, _, e, _ = h.estimate(img)
        vis.clear()
        if p is not None:
            p = p - p[0]
            vis.print_pose(p, e)
        vis.draw()
        vis.sleep(0.001)
        # cv2.imshow("", img)
        # cv2.waitKey(1)
