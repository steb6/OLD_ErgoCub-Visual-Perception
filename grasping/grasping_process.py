import copy
import sys
import time
from multiprocessing import Queue, Process
from queue import Empty

import cv2
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation
from vispy import scene
from vispy.scene import Markers

from grasping.modules.ransac.forge.fit_plane_speed import plot_plane
from grasping.modules.seg_pcr_ge.eval.output import plot_line
from grasping.modules.utils.input import RealSense
from grasping.modules.utils.timer import Timer
from utils.concurrency import Node
from utils.visualization import draw_geometries, draw_geometries_img

logger.remove()
logger.add(sys.stdout,
           format="<fg #b28774>{time:YYYY-MM-DD HH:mm:ss:SSS ZZ}</> <yellow>|</>"
                  " <lvl>{level: <8}</> "
                  "<yellow>|</> <blue>{process.name: ^12}</> <yellow>-</> <lvl>{message}</>",
           diagnose=True)

logger.level('INFO', color='<fg #fef5ed>')
logger.level('SUCCESS', color='<fg #79d70f>')
logger.level('WARNING', color='<fg #fd811e>')
logger.level('ERROR', color='<fg #ed254e>')


class Grasping(Node):
    def __init__(self):
        super().__init__(name='grasping')

    def startup(self):
        import pycuda.autoinit
        import torch
        from grasping.modules.denoising.src.denoiser import Denoising
        from grasping.modules.ransac.utils.inference import Runner
        from grasping.modules.shape_reconstruction.tensorrt.utils.inference import Infer as InferPcr

        a = torch.zeros([1]).to('cuda')
        logger.info('Loading Shape Reconstruction engine')
        self.backbone = InferPcr('grasping/modules/shape_reconstruction/tensorrt/assets/pcr.engine')
        logger.success('Shape Reconstruction engine loaded')

        from grasping.modules.segmentation.tensorrt.utils.inference import Infer as InferSeg

        logger.info('Loading segmentation engine')
        self.model = InferSeg('./grasping/modules/segmentation/tensorrt/assets/seg_int8.engine')
        logger.success('Segmentation engine loaded')

        from grasping.modules.seg_pcr_ge.delete import GraspEstimator
        # from ransac.utils.grasp_estimator import GraspEstimator

        logger.info('Loading RANSAC engine')
        self.ransac = Runner('./grasping/modules/ransac/assets/ransac_5000.engine')
        logger.success('RANSAC engine loaded')

        from grasping.modules.shape_reconstruction.tensorrt.utils.decoder import Decoder

        self.decoder = Decoder()

        self.grasp_estimator = GraspEstimator(self.ransac)
        self.denoising = Denoising()

        self._out_queues['human'] = self.manager.get_queue('grasping_human')
        self._out_queues['visualizer'] = self.manager.get_queue('grasping_visualizer')

    def loop(self, data):
        # Outputs Standard Values
        hands = None
        center = None
        denormalize = None
        mean = 0
        var = 1
        res = None
        normalized_pc = None

        # Input
        rgb = data['rgb']
        depth = data['depth']

        # Setup
        R = Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
        flip_z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        # Pipeline
        mask = self.model(rgb)
        mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

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
            #   Z axis symmetry, std scaling, mean translation, 180dg rotation
            denormalize = compose_transformations([flip_z, np.eye(3) * (var * 2), mean[np.newaxis], R])

            # Reconstruction
            fast_weights = self.backbone(normalized_pc)
            res = self.decoder(fast_weights)

            if res.shape[0] < 10_000:
                center = np.mean((np.block([res, np.ones([res.shape[0], 1])]) @ denormalize)[..., :3], axis=0)[None]

                poses = self.grasp_estimator.find_poses(res @ flip_z, 0.001, 5000)

                if poses is not None:
                    hands = {}
                    hands['right'] = compose_transformations([poses[1].T, poses[0][np.newaxis] * (var * 2) + mean, R])
                    hands['left'] = compose_transformations([poses[3].T, poses[2][np.newaxis] * (var * 2) + mean, R])
                else:
                    logger.warning('Couldn\'t generate hand poses')
            else:
                logger.warning('Corrupted reconstruction - check the input point cloud')
                res = None
        else:
            logger.warning('Warning: not enough input points. Skipping reconstruction')

        output = {}

        output['human'] = {'hands': hands, 'center': center, 'mask': mask}

        if 'debug' in data and data['debug']:
            o3d_scene = RealSense.rgb_pointcloud(depth, rgb)
            output['visualizer'] = {'rgb': rgb, 'depth': depth, 'mask': mask, 'distance': distance, 'partial': normalized_pc,
             'scene': np.concatenate([np.array(o3d_scene.points) @ R, np.array(o3d_scene.colors)], axis=1),
             'reconstruction': res, 'hands': hands,
             'transform': denormalize}

        return output

    def shutdown(self):
        pass


def compose_transformations(tfs):
    ''''All 3x3 matrices are padded with an additional row and column from the Identity Matrix
        All the 1x3 matrices are'''
    c = np.eye(4)

    for t in tfs:
        if not isinstance(t, np.ndarray):
            raise ValueError('Transformations must be numpy.ndarray.')

        if t.shape == (3, 3):
            c = c @ np.block([[t, np.zeros([3, 1])],
                              [np.zeros([1, 3]), np.ones([1, 1])]])
        elif t.shape == (1, 3):
            c = c @ np.block([[np.eye(3), np.zeros([3, 1])],
                              [t, np.ones([1, 1])]])
        else:
            raise ValueError(f'Shape {t.shape} not allowed.')

    return c

if __name__ == '__main__':
    grasping = Grasping()
    grasping.run()