import time

import torch
from polygraphy.backend.onnx import OnnxFromPath, GsFromOnnx
from polygraphy.backend.onnxrt import SessionFromOnnx, OnnxrtRunner
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner, EngineFromBytes, CreateConfig
from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader, Comparator
import numpy as np
import onnxruntime as ort

if __name__ == '__main__':

    onnx_file = './shape_reconstruction/tensorrt/assets/noise_pcr.onnx'
    engine_file = './shape_reconstruction/tensorrt/assets/noise_pcr.engine'

    data_loader = DataLoader(iterations=100,
                             val_range=(-0.5, 0.5),
                             input_metadata=TensorMetadata.from_feed_dict({'input': np.zeros([1, 3, 192, 256], dtype=np.float32)}))

    # config = CreateConfig(fp16=True, tf32=True)
    config = CreateConfig()

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config)
    engine = build_engine()

    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())
