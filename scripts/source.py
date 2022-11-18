import copy

from loguru import logger

from configs.source_config import Logging, Network, Input
from utils.concurrency import SrcNode, SrcYarpNode
from utils.logging import setup_logger

setup_logger(level=Logging.level)


@logger.catch(reraise=True)
class Source(SrcYarpNode):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.camera = None

    def startup(self):
        self.camera = Input.camera(**Input.Params.to_dict())

    def loop(self):

        while True:
            try:
                rgb, depth = self.camera.read()
                data = {'rgb': copy.deepcopy(rgb), 'depth': copy.deepcopy(depth)}

                return {k: data for k in Network.Args.out_queues}

            except RuntimeError as e:
                logger.error("Realsense: frame didn't arrive")
                self.camera = Input.camera(**Input.Params.to_dict())


if __name__ == '__main__':
    source = Source()
    source.run()