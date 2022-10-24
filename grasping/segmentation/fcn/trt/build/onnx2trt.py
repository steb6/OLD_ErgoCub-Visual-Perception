from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, CreateConfig


def main():
    onnx_file = 'grasping/modules/segmentation/tensorrt/assets/segmentation.onnx'
    engine_file = 'grasping/modules/segmentation/tensorrt/assets/seg_fp16_docker.engine'

    config = CreateConfig(fp16=True, tf32=True)

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config)
    engine = build_engine()

    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())


if __name__ == '__main__':
    main()
