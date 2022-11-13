import copy

from loguru import logger

from configs.source_config import Logging, Network, Input
from utils.concurrency import SrcYarpNode
from utils.logging import setup_logger

setup_logger(**Logging.Logger.Params.to_dict())


@logger.catch(reraise=True)
class Source(SrcYarpNode):
    def __init__(self):
        super().__init__(**Network.to_dict())
        self.camera = None

        self.i = 0
        self.rgb, self.depth = None, None

    def startup(self):
        self.camera = Input.camera(**Input.Params.to_dict())

    def loop(self):

        while True:
            try:
                while self.i != 261:
                    self.rgb, self.depth = self.camera.read()
                    self.i += 1

                logger.info("Sending frame.", recurring=True)
                data = {'rgb': copy.deepcopy(self.rgb), 'depth': copy.deepcopy(self.depth)}
                return {k: data for k in Network.out_queues}

            except RuntimeError as e:
                logger.error("Realsense: frame didn't arrive")
                self.camera = Input.camera(**Input.Params.to_dict())


if __name__ == '__main__':
    source = Source()
    source.run()