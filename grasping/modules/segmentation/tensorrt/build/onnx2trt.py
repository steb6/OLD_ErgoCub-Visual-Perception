import time
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from polygraphy.backend.onnx import OnnxFromPath, GsFromOnnx
from polygraphy.backend.onnxrt import SessionFromOnnx, OnnxrtRunner
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner, EngineFromBytes, CreateConfig, \
    Calibrator
from polygraphy.common import TensorMetadata
from polygraphy.comparator import Comparator
import numpy as np
# import onnxruntime as ort
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import tensorrt as trt


def main():
    onnx_file = 'grasping/modules/segmentation/tensorrt/assets/seg_test.onnx'
    engine_file = 'grasping/modules/segmentation/tensorrt/assets/seg_fp16_docker.engine'

    # dataloader = DataLoader(iterations=22)
    config = CreateConfig(fp16=True, tf32=True) # , int8=True, calibrator=Calibrator(dataloader)

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config)
    engine = build_engine()

    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())


class DataLoader:

    def __init__(self, iterations):
        self.iterations = iterations
        self.root = Path('./segmentation/tensorrt/assets/real_data')

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

        res = OrderedDict()
        res['input'] = self.items[i].astype(trt.nptype(trt.float32)).ravel()
        return res

    def __len__(self):
        return self.iterations

if __name__ == '__main__':
    # dl = DataLoader(iterations=22)
    #
    # for x in dl:
    #     print(x[''])
    main()