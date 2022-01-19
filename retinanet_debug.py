from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
from torch import Tensor

try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone,  # torchvision==0.10.0
    _validate_trainable_layers,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection import RetinaNet

model_urls = {
    "retinanet_resnet50_fpn_coco": "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
}

box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))


def postprocess_detections(
    head_outputs,  # type: Dict[str, List[Tensor]]
    anchors,  # type: List[List[Tensor]]
    image_shapes,  # type: List[Tuple[int, int]]
    score_thresh=0.05,
    topk_candidates=1000,
    nms_thresh=0.5,
    detections_per_img=300,
) -> List[Dict[str, Tensor]]:
    class_logits = head_outputs["cls_logits"]
    box_regression = head_outputs["bbox_regression"]

    num_images = len(image_shapes)

    detections: List[Dict[str, Tensor]] = []

    for index in range(num_images):
        box_regression_per_image = [br[index] for br in box_regression]
        logits_per_image = [cl[index] for cl in class_logits]
        anchors_per_image, image_shape = anchors[index], image_shapes[index]

        image_boxes = []
        image_scores = []
        image_labels = []

        for box_regression_per_level, logits_per_level, anchors_per_level in zip(
            box_regression_per_image, logits_per_image, anchors_per_image
        ):
            num_classes = logits_per_level.shape[-1]

            # remove low scoring boxes
            scores_per_level = torch.sigmoid(logits_per_level).flatten()
            keep_idxs = scores_per_level > score_thresh
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # keep only topk scoring predictions
            num_topk = min(topk_candidates, topk_idxs.size(0))
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            boxes_per_level = box_coder.decode_single(
                box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
            )
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, nms_thresh)
        keep = keep[:detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
            }
        )

    return detections


class MyRetinaNet(RetinaNet):
    def __init__(
        self,
        backbone,
        num_classes,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=1000,
    ):
        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # Anchor parameters
            anchor_generator,
            head,
            proposal_matcher,
            score_thresh,
            nms_thresh,
            detections_per_img,
            fg_iou_thresh,
            bg_iou_thresh,
            topk_candidates,
        )
        self.transform = None

    @staticmethod
    def get_original_image_sizes(images: List[Tensor]):
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        return original_image_sizes

    def forward(self, images, targets=None):
        assert (
            self.training is False
        ), "training mode not supported in this customized RetinaNet!"

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        # get the features from the backbone
        features = self.backbone(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs["cls_logits"].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        # split outputs per level
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(
                head_outputs[k].split(num_anchors_per_level, dim=1)
            )

        return features, split_head_outputs, num_anchors_per_level


def my_retinanet_resnet50_fpn(
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    **kwargs,
):
    # ---------- ↓ Same as retinanet_resnet50_fpn ----------

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone(
        "resnet50",
        pretrained_backbone,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
        trainable_layers=trainable_backbone_layers,
    )

    # ---------- ↑ Same as retinanet_resnet50_fpn ----------

    model = MyRetinaNet(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["retinanet_resnet50_fpn_coco"], progress=progress
        )
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model


if __name__ == "__main__":
    min_size = 800
    max_size = 1333
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    # model = MyRetinaNet()
    model = my_retinanet_resnet50_fpn(pretrained=True).eval()

    # images, targets = data
    images = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    targets = None

    # get the original image sizes
    original_image_sizes = MyRetinaNet.get_original_image_sizes(images)
    trans_images, trans_targets = transform(
        images, targets
    )  # Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]

    features, split_head_outputs, num_anchors_per_level = model(trans_images.tensors)

    # create the set of anchors
    anchors = model.anchor_generator(trans_images, features)
    split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

    # compute the detections
    # detections: List[Dict[str, Tensor]]
    detections = postprocess_detections(
        split_head_outputs, split_anchors, trans_images.image_sizes
    )
    detections = transform.postprocess(
        detections, trans_images.image_sizes, original_image_sizes
    )

