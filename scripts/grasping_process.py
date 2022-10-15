import copy
import time

import cv2
import numpy as np
# from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from grasping.modules.ransac.build.test import test
# from sklearn.neighbors import NearestNeighbors

# from grasping.modules.tracking.icp import icp
from grasping.modules.utils.input import RealSense
from grasping.modules.utils.misc import draw_mask
from gui.misc import project_pc, project_hands
from utils.concurrency import Node

from utils.logging import get_logger
import pycuda.autoinit
from grasping.modules.denoising.src.denoiser import Denoising
from grasping.modules.ransac.utils.inference import TRTRunner as Runner
from grasping.modules.shape_reconstruction.tensorrt.utils.inference import TRTRunner as InferPcr
from grasping.modules.segmentation.tensorrt.utils.inference import TRTRunner as InferSeg
from grasping.modules.seg_pcr_ge.delete import GraspEstimator
from grasping.modules.shape_reconstruction.tensorrt.utils.decoder import Decoder

logger = get_logger(True)


class Grasping(Node):
    def __init__(self):
        super().__init__(name='grasping')

    def startup(self):

        self.backbone = InferPcr('grasping/modules/shape_reconstruction/tensorrt/assets/pcr_docker.engine')


        logger.info('Loading segmentation engine...')
        self.model = InferSeg('./grasping/modules/segmentation/tensorrt/assets/seg_fp16_docker.engine')
        logger.success('Segmentation engine loaded')

        logger.info('Loading RANSAC engine...')
        self.ransac = Runner('./grasping/modules/ransac/assets/ransac_5000_docker.engine')
        logger.success('RANSAC engine loaded')

        from grasping.modules.ransac.utils.inference import TRTRunner
        # test(self.ransac)

        self.decoder = Decoder()

        self.grasp_estimator = GraspEstimator(self.ransac)
        self.denoising = Denoising()

        self._out_queues['human'] = self.manager.get_queue('grasping_human')
        self._out_queues['visualizer'] = self.manager.get_queue('grasping_visualizer')
        self._out_queues['sink'] = self.manager.get_queue('grasping_sink')

        self.fps_s = []
        self.max_partial_points = 0

        self.reconstruction = None
        self.prev_partial = None
        self.prev_denormalize = None

        # test(self.ransac)

    def loop(self, data):
        # Outputs Standard Values
        hands = None
        center = None
        denormalize = None
        mean = 0
        var = 1
        res = None
        normalized_pc = None

        start = time.perf_counter()
        # Input
        rgb = data['rgb']
        depth = data['depth']

        # Setup
        R = Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
        flip_z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        # Pipeline
        # rgb = cv2.resize(rgb, (640, int(640 * (rgb.shape[0] / rgb.shape[1]))))
        # rgb = np.concatenate([np.zeros([60, 640, 3]),rgb,np.zeros([60, 640, 3])])

        # depth = cv2.resize(depth, (640, int(640 * (depth.shape[0] / depth.shape[1]))))
        # depth = np.concatenate([np.zeros([60, 640]), depth, np.zeros([60, 640])])
        # depth = depth.astype(np.uint16)

        mask = self.model(rgb)
        mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

        # rgb = rgb.astype(np.uint8)

        segmented_depth = copy.deepcopy(depth)
        segmented_depth[mask != 1] = 0

        # Adjust size
        distance = segmented_depth[segmented_depth != 0].mean()
        if len(segmented_depth.nonzero()[0]) >= 4096:
            segmented_pc = RealSense.depth_pointcloud(segmented_depth)


            # Downsample
            idx = np.random.choice(segmented_pc.shape[0], 4096, replace=False)
            downsampled_pc = segmented_pc[idx]

            # Denoise
            denoised_pc = self.denoising(downsampled_pc)

            # Check tracking TODO move the check to the other tracking if
            # tracking = False
            # if denoised_pc.shape[0] > self.max_partial_points:
            #     self.max_partial_points = denoised_pc.shape[0]
            #     logger.success('Increased visibility')
            # else:
            #     tracking = True
            #     logger.info('Tracking previous reconstruction')

            # Fix Size
            if denoised_pc.shape[0] > 2024:
                idx = np.random.choice(denoised_pc.shape[0], 2024, replace=False)
                size_pc = denoised_pc[idx]
            else:
                logger.warning('Info: Partial Point Cloud padded')
                diff = 2024 - denoised_pc.shape[0]
                pad = np.zeros([diff, 3])
                pad[:] = segmented_pc[0]
                size_pc = np.vstack((denoised_pc, pad))

            # Normalize
            mean = np.mean(size_pc, axis=0)
            var = np.sqrt(np.max(np.sum((size_pc - mean) ** 2, axis=1)))
            normalized_pc = (size_pc - mean) / (var * 2)
            normalized_pc[..., -1] = -normalized_pc[..., -1]
            # Set-up the inverse transformation

            # if False and tracking and self.reconstruction is not None and self.prev_partial is not None:
            #     # Option 1 track object_model and size_pc
            #     # Option 2 track previous size_pc and size_pc
            #     icp_transformation = icp(self.prev_partial, size_pc)
            #     # self.object_model = (np.block([self.object_model, np.ones([self.object_model.shape[0], 1])]) @ icp_transformation.T)[..., :3]
            #     self.prev_denormalize = compose_transformations([self.prev_denormalize, icp_transformation.T])
            #     denormalize = compose_transformations([self.prev_denormalize, R])
            #
            #     a = (np.block([self.reconstruction, np.ones([self.reconstruction.shape[0], 1])]) @ denormalize)[:, :3]
            #     b = size_pc @ R
            #     # Y = cdist(a, b, 'euclidean')
            #     nbrs = NearestNeighbors(n_neighbors=1).fit(a)
            #     distances, _ = nbrs.kneighbors(b)
            #     if np.mean(distances) > 0.005:
            #         self.max_partial_points = 0
            # else:
            #   Z axis symmetry, std scaling, mean translation, 180dg rotation
            self.prev_denormalize = compose_transformations([flip_z, np.eye(3) * (var * 2), mean[np.newaxis]])
            denormalize = compose_transformations([self.prev_denormalize, R])

            # Reconstruction
            fast_weights = self.backbone(normalized_pc)
            self.reconstruction = self.decoder(fast_weights)

            if self.reconstruction.shape[0] < 10_000:
                center = np.mean((np.block([self.reconstruction, np.ones([self.reconstruction.shape[0], 1])]) @ denormalize)[..., :3], axis=0)[None]

                poses = self.grasp_estimator.find_poses(self.reconstruction @ flip_z, 0.001, 5000)

                if poses is not None:
                    hands = {}
                    hands['right'] = compose_transformations([poses[1].T, poses[0][np.newaxis] * (var * 2) + mean, R])
                    hands['left'] = compose_transformations([poses[3].T, poses[2][np.newaxis] * (var * 2) + mean, R])
                else:
                    logger.warning('Couldn\'t generate hand poses')
            else:
                logger.warning('Corrupted reconstruction - check the input point cloud')

            self.prev_partial = size_pc
        else:
            logger.warning('Warning: not enough input points. Skipping reconstruction')

        output = {}

        output['human'] = {'hands': hands, 'center': center, 'mask': mask}

        if 'debug' in data and data['debug']:
            o3d_scene = RealSense.rgb_pointcloud(depth, rgb)
            output['visualizer'] = {'rgb': rgb, 'depth': depth, 'mask': mask, 'distance': distance, 'partial': normalized_pc,
             'scene': np.concatenate([np.array(o3d_scene.points) @ R, np.array(o3d_scene.colors)], axis=1),
             'reconstruction': self.reconstruction, 'hands': hands,
             'transform': denormalize}

        # # Compute fps
        end = time.perf_counter()
        self.fps_s.append(1. / (end - start) if (end - start) != 0 else 0)
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)

        # Light visualizer
        img = rgb
        if center is not None:
            img = cv2.circle(img, project_pc(center)[0], 5, (0, 255, 0)).astype(np.uint8)

        if mask is not None:
            img = draw_mask(img, mask)

        if hands is not None:
            img = project_hands(img, hands['right'], hands['left'])

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.putText(img, "FPS: {}".format(int(fps)), (10, 20), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 1,
                          cv2.LINE_AA)

        output = {'sink': {'img': img}}

        return output

    def shutdown(self):
        pass


def compose_transformations(tfs):
    """'All 3x3 matrices are padded with an additional row and column from the Identity Matrix
        All the 1x3 matrices are"""
    c = np.eye(4)

    for t in tfs:
        if not isinstance(t, np.ndarray):
            raise ValueError('Transformations must be numpy.ndarray.')

        if t.shape == (4, 4):
            pass
        elif t.shape == (3, 3):
            t = np.block([[t, np.zeros([3, 1])],
                              [np.zeros([1, 3]), np.ones([1, 1])]])
        elif t.shape == (1, 3):
            t = np.block([[np.eye(3), np.zeros([3, 1])],
                              [t, np.ones([1, 1])]])
        else:
            raise ValueError(f'Shape {t.shape} not allowed.')

        c = c @ t

    return c

if __name__ == '__main__':
    grasping = Grasping()
    grasping.run()