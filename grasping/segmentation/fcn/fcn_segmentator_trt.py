from grasping.utils.inference import TRTRunner


class FcnSegmentatorTRT:
    def __init__(self, engine_path):
        self.engine = TRTRunner(engine_path)

    def __call__(self, x):
        res = self.engine(x)
        res = res[0].reshape(192, 256, 1)

        return res
