# Infer imagenet with torchvision
#
# Usage:
#
# python quantize_mobilenet_v3.py path/to/imagenet mobilenetv3.mobilenet_v3_large --device cuda --bsize=50
#
# python quantize_mobilenet_v3.py path/to/imagenet quantized_mobilenetv3  --bsize=50

import argparse

import torch
import torch.nn as nn
import torch.quantization
import torchvision
from torch import Tensor
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torchvision import datasets
from torchvision.models.mobilenetv3 import (
    InvertedResidual,
    ConvBNActivation,
    MobileNetV3,
    SqueezeExcitation,
    _mobilenet_v3_conf,
)
from torchvision.models.quantization.utils import _replace_relu
import torchvision.transforms as transforms


model_names = sorted(
    name
    for name in torchvision.models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision.models.__dict__[name])
)


class QuantizableSqueezeExcitation(SqueezeExcitation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, input: Tensor) -> Tensor:
        return self.skip_mul.mul(self._scale(input, False), input)

    def fuse_model(self):
        fuse_modules(self, ["fc1", "relu"], inplace=True)


class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, se_layer=QuantizableSqueezeExcitation, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.block(x))
        else:
            return self.block(x)


class QuantizableMobileNetV3(MobileNetV3):
    def __init__(self, *args, **kwargs):
        """
        MobileNet V3 main class
        Args:
           Inherits args from floating point MobileNetV3
        """
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNActivation:
                modules_to_fuse = ["0", "1"]
                if type(m[2]) == nn.ReLU:
                    modules_to_fuse.append("2")
                fuse_modules(m, modules_to_fuse, inplace=True)
            elif type(m) == QuantizableSqueezeExcitation:
                m.fuse_model()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, neval_batches, device="cpu"):
    # batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(data_loader), [top1, top5], prefix="Test: ")

    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            target = target.to(device)
            image = image.to(device)
            output = model(image)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if i % 50 == 0:
                progress.display(i)

            if i >= neval_batches:
                return top1, top5

    return top1.avg, top5.avg


def load_model(model_file, arch):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch)
    model = QuantizableMobileNetV3(
        inverted_residual_setting,
        last_channel,
        block=QuantizableInvertedResidual,
    )
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to("cpu")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="path to validation data",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=model_names,
        help="model name",
    )
    parser.add_argument(
        "--bsize",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device",
    )
    args = parser.parse_args()

    arch = args.arch
    assert arch in ["mobilenet_v3_small", "mobilenet_v3_large"]
    data_path = args.data

    eval_batch_size = args.bsize  # 50000 images in total
    num_eval_batches, r = divmod(50000, args.bsize)
    assert r == 0, "barch size must be divisible."

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            data_path,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    print("Float model evaluation")
    float_model = torchvision.models.__dict__[arch](pretrained=True)
    # float_model.load_state_dict(torch.load(float_model_file))
    float_model.to(args.device)
    top1, top5 = evaluate(float_model, val_loader, num_eval_batches, args.device)
    print(
        "Evaluation accuracy on {n} images\nAcc@1 {top1:.3f} Acc@5 {top5:.3f}".format(
            n=num_eval_batches * eval_batch_size, top1=top1, top5=top5
        )
    )

    # 4. Post-training static quantization
    # ------------------------------------

    num_calib_batches = num_eval_batches

    # TODO: do not hard code the pretrained weight path
    float_model_file = "/path/to/mobilenet_v3_large-8738ca79.pth"
    per_channel_quantized_model = load_model(float_model_file, arch)
    _replace_relu(per_channel_quantized_model)
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig(
        "qnnpack"
    )
    print(per_channel_quantized_model.qconfig)

    torch.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model, val_loader, num_calib_batches)
    torch.quantization.convert(per_channel_quantized_model, inplace=True)
    top1, top5 = evaluate(
        per_channel_quantized_model,
        val_loader,
        neval_batches=num_calib_batches,
    )
    print(
        "Evaluation accuracy on {n} images\nAcc@1 {top1:.3f} Acc@5 {top5:.3f}".format(
            n=50000, top1=top1, top5=top5
        )
    )


if __name__ == "__main__":
    main()
