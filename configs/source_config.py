from logging import INFO

from utils.confort import BaseConfig
from utils.input import RealSense
import pyrealsense2 as rs


class Logging(BaseConfig):
    level = INFO


class Network(BaseConfig):
    ip = 'localhost'
    port = 50000
    out_queues = ['source_grasping', 'source_action_rec']
    # make the output queue blocking (can be used to put a breakpoint in the sink and debug the process output)
    blocking = False


class Input(BaseConfig):
    camera = RealSense

    class Params:
        rgb_res = (640, 480)
        depth_res = (640, 480)
        fps = 30
        depth_format = rs.format.z16
        color_format = rs.format.rgb8
        from_file = 'assets/test_640.bag'
        skip_frames = False
