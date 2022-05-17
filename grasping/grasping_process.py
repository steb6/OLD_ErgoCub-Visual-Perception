import copy
import time
from multiprocessing import Queue, Process
from queue import Empty

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from grasping.modules.utils.input import RealSense
from grasping.modules.utils.misc import draw_mask
from grasping.modules.utils.timer import Timer
from utils.concurrency import Node
from utils.output2 import VISPYVisualizer


class Grasping(Node):
    def __init__(self):
        super().__init__(name='grasping')

    def startup(self):
        import pycuda.autoinit
        import torch
        torch.set_num_threads(1)
        from grasping.modules.denoising.src.denoiser import Denoising
        from grasping.modules.ransac.utils.inference import Runner
        from grasping.modules.shape_reconstruction.tensorrt.utils.inference import Infer as InferPcr

        a = torch.zeros([1]).to('cuda')
        print('Loading Shape Reconstruction engine')
        self.backbone = InferPcr('grasping/modules/shape_reconstruction/tensorrt/assets/pcr.engine')
        print('Shape Reconstruction engine loaded')

        from grasping.modules.segmentation.tensorrt.utils.inference import Infer as InferSeg

        print('Loading segmentation engine')
        self.model = InferSeg('./grasping/modules/segmentation/tensorrt/assets/seg_int8.engine')
        print('Segmentation engine loaded')

        from grasping.modules.seg_pcr_ge.delete import GraspEstimator
        # from ransac.utils.grasp_estimator import GraspEstimator

        # print('Loading RANSAC engine')
        self.ransac = Runner('./grasping/modules/ransac/assets/ransac_5000.engine')
        # print('RANSAC engine loaded')

        from grasping.modules.shape_reconstruction.tensorrt.utils.decoder import Decoder

        self.decoder = Decoder()

        self.grasp_estimator = GraspEstimator(self.ransac)
        self.denoising = Denoising()

        self.out_queue = self.manager.get_queue('grasping_out')

        self.verbose = False
        self.last_time = 0
        self.fps_s = []


    def loop(self, data):
        start = time.perf_counter()

        rgb = data['rgb']
        depth = data['depth']

        with Timer('segmentation'):
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
            #
            # global i
            # np.save(f'pc_noise{i}', downsampled_pc)
            # i += 1
            # if i == 10:
            #     exit()

            with Timer(name='denoise'):
                # Denoise
                # clustering = DBSCAN(eps=0.05, min_samples=10).fit(downsampled_pc)  # 0.1 10 are perfect but slow
                # close = clustering.labels_[downsampled_pc.argmax(axis=0)[2]]
                # denoised_pc = downsampled_pc[clustering.labels_ == close]

                denoised_pc = self.denoising(downsampled_pc)

            # denoised_pc = downsampled_pc

            if denoised_pc.shape[0] > 2024:
                idx = np.random.choice(denoised_pc.shape[0], 2024, replace=False)
                size_pc = denoised_pc[idx]
            else:
                if self.verbose:
                    print('Info: Partial Point Cloud padded')
                diff = 2024 - denoised_pc.shape[0]
                pad = np.zeros([diff, 3])
                pad[:] = segmented_pc[0]
                size_pc = np.vstack((denoised_pc, pad))


            # Normalize
            mean = np.mean(size_pc, axis=0)
            var = np.sqrt(np.max(np.sum((size_pc - mean) ** 2, axis=1)))
            normalized_pc = (size_pc - mean) / (var * 2)
            normalized_pc[..., -1] = -normalized_pc[..., -1]

            with Timer(name='backbone'):
                # Reconstruction
                fast_weights = self.backbone(normalized_pc)

                # global i
                # np.save(f'test{i}', normalized_pc)
                # i += 1
                # if i == 10:
                #     exit()

            with Timer(name='implicit function'):
                res = self.decoder(fast_weights)
                if self.verbose:
                    print(res.shape[0])

            if res.shape[0] < 10_000:
                poses = self.grasp_estimator.find_poses(res * np.array([1, 1, -1]), 0.001, 5000)
                # poses = None
                if poses is not None:
                    poses[0] = (poses[0] * (var * 2) + mean)
                    poses[2] = (poses[2] * (var * 2) + mean)
            else:
                if self.verbose:
                    print('Warning: corrupted results. Probable cause: too much input noise')
                poses = None
                mean = 0
                var = 1
                res = np.array([[0, 0, 0]])
                normalized_pc = np.array([[0, 0, 0]])
        else:
            if self.verbose:
                print('Warning: not enough input points. Skipping reconstruction')
            poses = None
            mean = 0
            var = 1
            res = np.array([[0, 0, 0]])
            normalized_pc = np.array([[0, 0, 0]])

        # outputs = {'mask': mask, 'partial': size_pc, 'reconstruction': (res * np.array([1, 1, -1]) * (var * 2) + mean),
        #         'grasp_poses': poses, 'distance': distance}

        # Visualization
        if self.verbose:
            fps = 1 / (time.perf_counter() - start)
            print('\r')
            for k, v in Timer.counters.items():
                print(f'{k}: {1 / (Timer.timers[k] / v)}', end=' ')
            print(f'tot: {fps}', end=' ')

        if not self.out_queue.empty():
            try:
                self.out_queue.get(block=False)
            except Empty:
                pass

        R = Rotation.from_euler('x', 0, degrees=True).as_matrix()
        self.out_queue.put(np.mean((res * (var * 2) + mean * np.array([1, 1, -1])) @ R * np.array([1, -1, 1]), axis=0)[None, ...])

        o3d_scene = RealSense.rgb_pointcloud(depth, rgb)

        end = time.time()
        self.fps_s.append(1. / (end - self.last_time) if (end - self.last_time) != 0 else 0)
        self.last_time = end
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)
        print(fps)

        return {'rgb': rgb, 'mask': mask, 'distance': distance, 'partial': normalized_pc,
                'scene': np.concatenate([np.array(o3d_scene.points), np.array(o3d_scene.colors)], axis=1), 'reconstruction': res,
                'mean': mean, 'var': var}

    def shutdown(self):
        pass


if __name__ == '__main__':
    grasping = Grasping()
    grasping.run()