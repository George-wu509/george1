
import torch

載入模型

model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)

model.eval()

不同選項載入模型

model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=False)

查詢模型

torch.hub.list("pytorch/vision")

其他選項

dir(model)

help(model.forward)

Hub 下載的model file會放在.cache folder

[\\wsl.localhost\Ubuntu-20.04\home\georgewu\.cache\torch\hub\checkpoints](file://wsl.localhost/Ubuntu-20.04/home/georgewu/.cache/torch/hub/checkpoints)