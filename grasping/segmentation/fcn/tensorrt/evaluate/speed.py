import copy
import time
from multiprocessing.managers import BaseManager

import numpy as np
import pycuda.autoinit

from grasping.modules.utils.misc import draw_mask
from grasping.modules.utils.timer import Timer
from utils.input import RealSense


def main(runtime):

    data_loader = DataSet(iterations=1000)

    if runtime == 'trt':
        ####  Setup TensorRT Engine
        backbone = Infer('grasping/modules/segmentation/tensorrt/assets/seg_int8.engine')
        ####  Run Evaluation
        for i, x in tqdm.tqdm(enumerate(data_loader)):
            res = backbone(x['input'])

    if runtime == 'pt':
        model = models.segmentation.fcn_resnet101(pretrained=False)
        model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model.load_state_dict(torch.load('grasping/modules/segmentation/pytorch/checkpoints/sym5/epoch23'), strict=False)
        model.eval()
        model.cuda()

        with torch.no_grad():
            for i, x in tqdm.tqdm(enumerate(data_loader)):
                res = model(torch.tensor(x['input'], device='cuda'))

import cv2
from utils.concurrency import Node
class Demo:

    def __init__(self):
        self.startup()

    def startup(self):
        import pycuda.autoinit
        import torch
        from grasping.modules.denoising.src.denoiser import Denoising
        from grasping.modules.ransac.utils.inference import Runner
        from grasping.modules.shape_reconstruction.tensorrt.utils.inference import Infer as InferPcr


        a = torch.zeros([1]).to('cuda')
        print('Loading Shape Reconstruction engine')
        backbone = InferPcr('grasping/modules/shape_reconstruction/tensorrt/assets/pcr.engine')
        print('Shape Reconstruction engine loaded')

        from grasping.modules.segmentation.tensorrt.utils.inference import Infer as InferSeg

        print('Loading segmentation engine')
        self.model = InferSeg('./grasping/modules/segmentation/tensorrt/assets/seg_int8.engine')
        print('Segmentation engine loaded')

        from grasping.modules.seg_pcr_ge.delete import GraspEstimator
        # from ransac.utils.grasp_estimator import GraspEstimator

        # print('Loading RANSAC engine')
        ransac = Runner('./grasping/modules/ransac/assets/ransac_5000.engine')
        # print('RANSAC engine loaded')

        from grasping.modules.shape_reconstruction.tensorrt.utils.decoder import Decoder

        decoder = Decoder()

        grasp_estimator = GraspEstimator(ransac)
        denoising = Denoising()

        ####  Setup TensorRT Engine

        ####  Run Evaluation
        # camera = RealSense(color_format=rs.format.rgb8)

        BaseManager.register('get_queue')
        manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
        manager.connect()

        self.q = manager.get_queue('speed')

    def loop(self, data):

        rgb = data['rgb']
        depth = data['depth']

        start = time.perf_counter()
        mask = copy.deepcopy(self.model(copy.deepcopy(rgb)))
        print('\r', 1 / (time.perf_counter() - start), end='')
        Timer.reset()
        mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

        segmented_depth = copy.deepcopy(depth)
        segmented_depth[mask != 1] = 0

        # Adjust size
        distance = segmented_depth[segmented_depth != 0].mean()

        if len(segmented_depth.nonzero()[0]) >= 4096:
            pass
        else:
            poses = None
            mean = 0
            var = 1
            res = np.array([[0, 0, 0]])
            normalized_pc = np.array([[0, 0, 0]])

        # outputs = {'mask': mask, 'partial': size_pc, 'reconstruction': (res * np.array([1, 1, -1]) * (var * 2) + mean),
        #         'grasp_poses': poses, 'distance': distance}

        # Visualization

        # fps = 1 / (time.perf_counter() - start)
        # print('\r')
        # for k, v in Timer.counters.items():
        #     print(f'{k}: {1 / (Timer.timers[k] / v)}', end=' ')
        # print(f'tot: {fps}', end=' ')

        # avg_fps = {name: 1 / Timer(name).compute() for name in Timer.timers}

        o3d_scene = RealSense.rgb_pointcloud(depth, rgb)
        #
        # Timer.reset()
        # return {'rgb': rgb, 'depth': depth, 'mask': mask, 'distance': distance, 'partial': normalized_pc,
        #         'reconstruction': res, 'poses': poses,
        #         'mean': mean, 'var': var}

    def run(self):
        while True:
            self.loop(self.q.get())
def demo():


    while True:
        # rgb, depth = camera.read()
        rgb = q.get()['rgb']
        with Timer('seg'):
            mask = model(rgb)
        mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

        print(1 / Timer('seg').compute())
        Timer.reset()
        res = draw_mask(rgb, mask)

        cv2.imshow('', res)
        cv2.waitKey(1)

if __name__ == '__main__':
    Demo().run()