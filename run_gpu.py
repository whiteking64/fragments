import time

import torchvision
import torch


x = torch.Tensor(4,3,224,224).cuda()
model = torchvision.models.resnet18().cuda()
while True:
    _ = model(x)
    time.sleep(1)

