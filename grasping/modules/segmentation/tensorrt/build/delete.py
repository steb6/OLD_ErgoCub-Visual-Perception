import torch
from torch import nn
from torchvision import models
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        model.eval()
        model.cuda()

        self.model = model

        self.tr = T.Compose([lambda x: x.unsqueeze(0).permute(0, 3, 1, 2),
                             lambda x: x / 255,
                             T.Resize((192, 256), InterpolationMode.BILINEAR),
                             T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
                                         std=[0.229, 0.224, 0.225])])

    def forward(self, x):
        x = self.tr(x)
        x = self.model(x)['out']
        x = torch.softmax(x, dim=1)
        x = torch.argmax(x, dim=1).permute([1, 2, 0])

        # x = T.Resize((480, 640), InterpolationMode.NEAREST)(x)
        return x


model = Model()
x = torch.randn((480, 640, 3)).cuda()

torch.onnx.export(model, x, './segmentation/tensorrt/assets/deeplab.onnx', input_names=['input'],
                  output_names=[f'output'], opset_version=11)
