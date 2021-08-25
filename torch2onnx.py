# template for onverting a pytorch model into an onnx model.

import torch
import torchvision


model = torchvision.models.resnet18(pretrained=True)
torch.onnx.export(
    model,
    torch.randn(1, 3, 224, 224),
    "./resnet18.onnx",
    export_params=True,
    # verbose=True,
    input_names=["input"],
    output_names=["output"],
)
