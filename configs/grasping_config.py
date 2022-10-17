import os
from pathlib import Path

from grasping.denoising import DbscanDenoiser
from grasping.grasp_detection import RansacGraspDetectorTRT
from grasping.segmentation.fcn.fcn_segmentator_trt import FcnSegmentatorTRT
from grasping.shape_reconstruction.confidence_pcr.decoder import ConfidencePCRDecoder
from grasping.shape_reconstruction.confidence_pcr.encoder import ConfidencePCRDecoderTRT
from utils.confort import BaseConfig


class Logging:
    log = True
    debug = True
    # key options:
    #   ['rgb', 'depth', 'mask', 'fps', 'center', 'hands', 'partial', 'scene', 'reconstruction', 'transform']
    keys = ['rgb', 'fps', 'mask', 'hands']


class Network(BaseConfig):
    ip = 'host.docker.internal'
    port = 50000
    in_queue = 'grasping'
    out_queues = ['grasping_human', 'grasping_sink']


class Segmentation(BaseConfig):
    model = FcnSegmentatorTRT

    class Args:
        engine_path = './grasping/segmentation/fcn/tensorrt/assets/seg_fp16_docker.engine'


class Denoiser(BaseConfig):
    model = DbscanDenoiser

    class Args:
        # DBSCAN parameters
        eps = 0.05
        min_samples = 10


class ShapeCompletion(BaseConfig):
    class Encoder:
        model = ConfidencePCRDecoderTRT

        class Args:
            engine_path = 'grasping/shape_reconstruction/confidence_pcr/tensorrt/assets/pcr_docker.engine'

    class Decoder:
        model = ConfidencePCRDecoder

        class Args:
            no_points = 10_000
            steps = 20
            thr = 0.5


class GraspDetection(BaseConfig):
    model = RansacGraspDetectorTRT

    class Args:
        engine_path = './grasping/grasp_detection/ransac/assets/ransac_5000_docker.engine'
        # RANSAC parameters
        tolerance = 0.001
        iterations = 5000
