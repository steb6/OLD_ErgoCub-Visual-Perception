import os
from pathlib import Path

run_process = True
docker = True
file = 'scripts/grasping_pipeline.py'


class Docker:
    image = 'ecub-env'
    name = 'ecub-grasping'
    options = ['-it', '--rm', '--gpus=all']
    volumes = [f'{Path(os.getcwd()).as_posix()}:/home/ecub']


class Logging:
    log = True
    debug = True
    queue = 'grasping_sink'
    # key options:
    #   ['rgb', 'depth', 'mask', 'fps', 'center', 'hands', 'partial', 'scene', 'reconstruction', 'transform']
    keys = ['rgb', 'fps', 'mask', 'hands']


class Network:
    ip = 'locahost' if not docker else 'host.docker.internal'
    port = 50000
    in_queue = 'grasping'
    out_queues = ['grasping_human'] + [Logging.queue]


class Segmentation:
    engine_path = './grasping/segmentation/fcn/tensorrt/assets/seg_fp16_docker.engine'


class Denoiser:
    # DBSCAN parameters
    eps = 0.05
    min_samples = 10


class ShapeCompletion:
    class Encoder:
        engine_path = 'grasping/shape_reconstruction/confidence_pcr/tensorrt/assets/pcr_docker.engine'

    class Decoder:
        no_points = 10_000
        steps = 20
        thr = 0.5


class GraspDetection:
    engine_path = './grasping/grasp_detection/ransac/assets/ransac_5000_docker.engine'

    # RANSAC parameters
    tolerance = 0.001
    iterations = 5000
