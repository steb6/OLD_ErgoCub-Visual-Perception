from logging import INFO

from utils.confort import BaseConfig
from utils.input import RealSense
import pyrealsense2 as rs


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

class Network(BaseConfig):
    ip = 'localhost'
    port = 50000
    out_queues = ['source_grasping']
    # make the output queue blocking (can be used to put a breakpoint in the sink and debug the process output)
    blocking = True


class Input(BaseConfig):
    camera = RealSense

    class Params:
        rgb_res = (640, 480)
        depth_res = (640, 480)
        fps = 30
        depth_format = rs.format.z16
        color_format = rs.format.rgb8
        from_file = 'assets/robo_arena_hole_filling_640.bag'
        skip_frames = False
