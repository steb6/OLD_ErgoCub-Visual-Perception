import copy
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from grasping.utils.input import RealSense
from grasping.utils.misc import compose_transformations, reload_package
from grasping.utils.avg_timer import Timer
from utils.concurrency import Node

from utils.logging import get_logger
import tensorrt as trt
# https://github.com/NVIDIA/TensorRT/issues/1945
import torch
import pycuda.autoinit
from grasping.denoising import DbscanDenoiser
from grasping.grasp_detection import RansacGraspDetectorTRT
from grasping.segmentation import FcnSegmentatorTRT
from grasping.shape_reconstruction import ConfidencePCRDecoderTRT
from grasping.shape_reconstruction import ConfidencePCRDecoder

from configs.main_config import Config
Config = Config.Grasping

logger = get_logger(True)

class Watch():
    def __init__(self):
        self.ft = defaultdict(lambda: 0)
    def check(self):
        from configs import main_config

        # while True:
        for file in Path('configs').glob('*'):
            mt = os.path.getmtime(file.as_posix())

            if mt != self.ft[file.name] and self.ft[file.name] != 0:
                reload_package(main_config)
                from configs import main_config
                global Config
                Config = main_config.Config.Grasping
                logger.success('Configuration reloaded')
            self.ft[file.name] = mt

class Grasping(Node):
    def __init__(self):
        super().__init__(**Config.Network.to_dict())
        self.seg_model = None
        self.denoiser = None
        self.pcr_encoder = None
        self.pcr_decoder = None
        self.grasp_detector = None

        self.fps_s = []
        self.max_partial_points = 0

        self.reconstruction = None
        self.prev_denormalize = None

        self.timer = Timer(window=10)
        self.watch = Watch()

    def startup(self):

        self.seg_model = FcnSegmentatorTRT(**Config.Segmentation.to_dict())
        self.denoiser = DbscanDenoiser(**Config.Denoiser.to_dict())
        self.pcr_encoder = ConfidencePCRDecoderTRT(**Config.ShapeCompletion.Encoder.to_dict())
        self.pcr_decoder = ConfidencePCRDecoder(**Config.ShapeCompletion.Decoder.to_dict())
        self.grasp_detector = RansacGraspDetectorTRT(**Config.GraspDetection.to_dict())

    @logger.catch(reraise=True)
    def loop(self, data):
        self.watch.check()

        output = {}

        self.timer.start()
        # Input
        rgb = data['rgb']
        depth = data['depth']

        if Config.Logging.debug:
            output['rgb'] = rgb
            output['depth'] = depth

        # Setup transformations
        R = Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
        flip_z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        # Segment the rgb and extract the object depth
        mask = self.seg_model(rgb)
        mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

        if Config.Logging.debug:
            output['mask'] = mask

        segmented_depth = copy.deepcopy(depth)
        segmented_depth[mask != 1] = 0

        if len(segmented_depth.nonzero()[0]) < 4096:
            logger.warning('Warning: not enough input points. Skipping reconstruction')
            output['fps'] = 1 / self.timer.compute(stop=True)
            output = {k: v for k, v, in output.items() if k in Config.Logging.keys}
            res = {'grasping_sink': output}
            return res

        distance = segmented_depth[segmented_depth != 0].mean()
        segmented_pc = RealSense.depth_pointcloud(segmented_depth)

        # Downsample
        idx = np.random.choice(segmented_pc.shape[0], 4096, replace=False)
        downsampled_pc = segmented_pc[idx]

        # Denoise
        denoised_pc = self.denoiser(downsampled_pc)

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

        self.prev_denormalize = compose_transformations([flip_z, np.eye(3) * (var * 2), mean[np.newaxis]])
        denormalize = compose_transformations([self.prev_denormalize, R])

        # Reconstruction
        fast_weights = self.pcr_encoder(normalized_pc)
        self.reconstruction = self.pcr_decoder(fast_weights)

        if self.reconstruction.shape[0] >= 10_000:
            logger.warning('Corrupted reconstruction - check the input point cloud')
            output['fps'] = 1 / self.timer.compute(stop=True)
            output = {k: v for k, v, in output.items() if k in Config.Logging.keys}
            res = {'grasping_sink': output}
            return res

        center = np.mean(
            (np.block(
                [self.reconstruction, np.ones([self.reconstruction.shape[0], 1])]) @ denormalize)[..., :3], axis=0
                        )[None]

        if Config.Logging.debug:
            output['center'] = center

        poses = self.grasp_detector(self.reconstruction @ flip_z)

        if poses is None:
            logger.warning('Corrupted reconstruction - check the input point cloud')
            output['fps'] = 1 / self.timer.compute(stop=True)
            output = {k: v for k, v, in output.items() if k in Config.Logging.keys}
            res = {'grasping_sink': output}
            return res

        hands = {'right': compose_transformations([poses[1].T, poses[0][np.newaxis] * (var * 2) + mean, R]),
                 'left': compose_transformations([poses[3].T, poses[2][np.newaxis] * (var * 2) + mean, R])}

        if Config.Logging.debug:
            output['hands'] = hands

        if Config.Logging.debug:
            o3d_scene = RealSense.rgb_pointcloud(depth, rgb)
            output['partial'] = normalized_pc
            output['scene'] = np.concatenate([np.array(o3d_scene.points) @ R, np.array(o3d_scene.colors)], axis=1)
            output['reconstruction'] = self.reconstruction
            output['transform'] = denormalize

        output['fps'] = 1 / self.timer.compute(stop=True)
        output = {k: v for k, v, in output.items() if k in Config.Logging.keys}
        res = {'grasping_sink': output}

        return res


if __name__ == '__main__':
    grasping = Grasping()
    grasping.run()