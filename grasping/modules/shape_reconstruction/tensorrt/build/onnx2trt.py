import time

import torch
from polygraphy.backend.onnx import OnnxFromPath, GsFromOnnx
from polygraphy.backend.onnxrt import SessionFromOnnx, OnnxrtRunner
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner, EngineFromBytes, CreateConfig
from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader, Comparator
import numpy as np
# import onnxruntime as ort
import tensorrt as trt

if __name__ == '__main__':

    onnx_file = 'grasping/modules/shape_reconstruction/tensorrt/assets/final.onnx'
    engine_file = 'grasping/modules/shape_reconstruction/tensorrt/assets/final_grasping.engine'

    # config = CreateConfig(fp16=True, tf32=True)
    config = CreateConfig(max_workspace_size=10000 << 40, profiling_verbosity=trt.ProfilingVerbosity.DETAILED)


    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config)
    engine = build_engine()  # ,profiles=profiles)

    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())
