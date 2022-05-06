import torch
from shape_reconstruction.pytorch.configs import server_config

from shape_reconstruction.pytorch.model import PCRNetwork



model = PCRNetwork.load_from_checkpoint('./shape_reconstruction/pytorch/checkpoint/noise_pcr', config=server_config.ModelConfig)
model.eval()
# model.cuda()


torch.onnx.export(model.backbone, torch.randn((1, 2024, 3)), './shape_reconstruction/tensorrt/assets/noise_pcr.onnx', input_names=['points'],
                  output_names=[f'output']  + [f'param{i}' for i in range(12)], opset_version=11) # [f'param{i}' for i in range(12)] + ['features']
# x = torch.randn((1, 3, 128)).cuda()
# torch.onnx.export(Simple(), x, 'production/test/simple.onnx', input_names=['input'], output_names=['idx'])