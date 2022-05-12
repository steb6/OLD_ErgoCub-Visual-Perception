import copy
import time
from multiprocessing import Queue, Process

import cv2
import numpy as np

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
        from grasping.modules.denoising.src.denoiser import Denoising
        from grasping.modules.ransac.utils.inference import Runner
        from grasping.modules.shape_reconstruction.tensorrt.utils.inference import Infer as InferPcr

        self.vis_queue = Queue(1)
        vis_proc = Process(target=VISPYVisualizer.create_visualizer,
                              args=(self.vis_queue,))
        vis_proc.start()

        a = torch.zeros([1]).to('cuda')
        print('Loading Shape Reconstruction engine')
        self.backbone = InferPcr('grasping/modules/shape_reconstruction/tensorrt/assets/final.engine')
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
                print(res.shape[0])

            if res.shape[0] < 10_000:
                poses = self.grasp_estimator.find_poses(res * np.array([1, 1, -1]), 0.001, 5000)
                # poses = None
                if poses is not None:
                    poses[0] = (poses[0] * (var * 2) + mean)
                    poses[2] = (poses[2] * (var * 2) + mean)
            else:
                print('Warning: corrupted results. Probable cause: too much input noise')
                poses = None
                mean = 0
                var = 1
                res = np.array([[0, 0, 0]])
                size_pc = np.array([[0, 0, 0]])
        else:
            print('Warning: not enough input points. Skipping reconstruction')
            poses = None
            mean = 0
            var = 1
            res = np.array([[0, 0, 0]])
            size_pc = np.array([[0, 0, 0]])

        outputs = {'mask': mask, 'partial': size_pc, 'reconstruction': (res * np.array([1, 1, -1]) * (var * 2) + mean),
                'grasp_poses': poses, 'distance': distance}

        mask, partial, reconstruction, poses, distance = \
            outputs['mask'], outputs['partial'], outputs['reconstruction'], outputs['grasp_poses'], outputs['distance']

        # Visualization

        res1, res2 = draw_mask(rgb, mask)

        font = cv2.FONT_ITALIC
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 2

        cv2.putText(res2, f'Distance: {distance / 1000}',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        self.vis_queue.put({'res1': cv2.flip(res1, 0), 'res2': cv2.flip(res2, 0), 'pc1': partial, 'pc2': reconstruction})

        fps = 1 / (time.perf_counter() - start)
        print('\r')
        for k, v in Timer.counters.items():
            print(f'{k}: {1 / (Timer.timers[k] / v)}', end=' ')
        print(f'tot: {fps}', end=' ')
        # cv2.imshow('Segmentation 1', cv2.cvtColor(res1, cv2.COLOR_RGB2BGR))
        # cv2.imshow('Segmentation 2', cv2.cvtColor(res2, cv2.COLOR_RGB2BGR))
        # cv2.imshow('Segmentation 1', res1)
        # cv2.imshow('Segmentation 2', res2)
        #
        # cv2.waitKey(1)


        # if poses is not None:
        #     best_centers = (poses[0], poses[2])
        #     best_rots = (poses[1], poses[3])
        #     size = 0.1
        # else:
        #     best_centers = (np.zeros([3]), np.zeros([3]))
        #     best_rots = (np.zeros([3, 3]), np.zeros([3, 3]))
        #     size = 0.01
        #
        # # Orient poses
        # for c, R, coord_mesh in zip(best_centers, best_rots, self.coords_mesh):
        #     coord_mesh_ = TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0]) \
        #         .rotate(R, center=[0, 0, 0]).translate(c, relative=False)
        #
        #     # Update mesh
        #     coord_mesh.triangles = coord_mesh_.triangles
        #     coord_mesh.vertices = coord_mesh_.vertices
        #
        # scene_pc = RealSense.rgb_pointcloud(depth, rgb)
        #
        # part_pc = PointCloud()
        # part_pc.points = Vector3dVector(partial)  # + [0, 0, 1]
        # part_pc.paint_uniform_color([0, 1, 0])
        # pred_pc = PointCloud()
        # pred_pc.points = Vector3dVector(reconstruction)
        # pred_pc.paint_uniform_color([1, 0, 0])
        #
        # self.scene_pcd.clear()
        # self.part_pcd.clear()
        # self.pred_pcd.clear()
        #
        # self.scene_pcd += scene_pc
        # self.part_pcd += part_pc
        # self.pred_pcd += pred_pc
        #
        # if not self.render_setup:
        #     self.vis2.add_geometry(self.scene_pcd)
        #     self.vis.add_geometry(self.part_pcd)
        #     self.vis.add_geometry(self.pred_pcd)
        #     for pose in self.coords_mesh:
        #         self.vis.add_geometry(pose)
        #
        #     render_setup = True
        #
        # self.vis2.update_geometry(self.scene_pcd)
        # self.vis.update_geometry(self.part_pcd)
        # self.vis.update_geometry(self.pred_pcd)
        # for pose in self.coords_mesh:
        #     self.vis2.update_geometry(pose)
        #
        # self.vis.poll_events()
        # self.vis.update_renderer()
        # self.vis2.poll_events()
        # self.vis2.update_renderer()

    def shutdown(self):
        pass