import numpy as np
import open3d as o3d # try to get rid of this


def depth_pointcloud(depth_image, intrinsics=None):
    depth_image = o3d.geometry.Image(depth_image)

    if intrinsics is None:
        intrinsics = {'fx': 612.7910766601562, 'fy': 611.8779296875, 'ppx': 321.7364196777344,
                      'ppy': 245.0658416748047, 'width': 640, 'height': 480}

    camera = o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'], intrinsics['fx'],
                                               intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, camera)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    return np.array(pcd.points)

import time
from pathlib import Path

from PIL import Image
from polygraphy.backend.onnx import OnnxFromPath, GsFromOnnx
from polygraphy.backend.onnxrt import SessionFromOnnx, OnnxrtRunner
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner, EngineFromBytes, CreateConfig, \
    Calibrator
from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader, Comparator
import numpy as np
# import torchvision.transforms as T
# from torchvision.transforms import InterpolationMode
import tensorrt as trt


if __name__ == '__main__':

    onnx_file = './assets/seg.onnx'
    engine_file = './assets/seg_int8.engine'

    config = CreateConfig(fp16=True, tf32=True, int8=True, calibrator=DataSet())

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config)
    engine = build_engine()

    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())


class DataSet:

    def __init__(self, iterations):
        self.iterations = iterations
        self.root = Path('./assets/calibration')

        tr = T.Compose([T.ToTensor(),
                        T.Pad([0, 80], fill=0, padding_mode='constant'),
                        T.Resize((512, 512), InterpolationMode.BILINEAR),
                        T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
                                    std=[0.229, 0.224, 0.225])])

        self.items = []
        for i in range(22):
            # self.items.append(tr(np.array(Image.open(self.root / f'ecub{i}_rgb.png'))).numpy()[None, ...])
            self.items.append(np.array(Image.open(self.root / f'ecub{i}_rgb.png')))

    def __getitem__(self, i):
        if i >= self.iterations:
            raise StopIteration

        i = i % 22
        return {'input': self.items[i].astype(trt.nptype(trt.float32)).ravel()}

    def __len__(self):
        return self.iterations
