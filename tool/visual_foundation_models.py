import torch
from torch import nn
from PIL import Image
from lavis.models import load_model_and_preprocess
import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from torchvision import transforms
import torch.nn.functional as F


class Resize(torch.nn.Module):
    def __init__(self, size, mode='bicubic', align_corners=False):
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)


class VisualFoundationModels(nn.Module):

    def __init__(self, device, prompt=None) -> None:
        super().__init__()
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.H, self.W = 364, 364
        self.device = device
        self.blip, self.blip_vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device)
        self.blip_prompt = "This is a scene of" if prompt is None else prompt
        self.preprocess = transforms.Compose(
            [
                Resize(size=(364, 364), mode='bicubic', align_corners=False),
                transforms.Normalize(mean, std)
            ]
        )
        self.captioner = self.blip
        self.detector = self.get_detector()
        pass

    def get_detector(self,):
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        self.total_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        predictor = DefaultPredictor(cfg)
        return predictor

    def get_caption(self, batch_input):
        # image = self.blip_vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        # print(image.shape)
        if batch_input.dtype == torch.uint8:
            batch_input = batch_input.float() / 255
        batch_input = self.preprocess(batch_input) # [B,C,H,W]→[B,C,364,364]
        batch_size = batch_input.shape[0]
        caption = self.captioner.generate({"image": batch_input.to(self.device), "prompt": [self.blip_prompt]*batch_size})
        return caption

    # def detect_object(self, raw_image):
    #     img_np = np.array(raw_image)
    #     outputs = self.detector(img_np[..., ::-1])
    #     pred_classes = outputs["instances"].pred_classes

    #     bbox = outputs["instances"].pred_boxes
    #     labels = [self.total_classes[i] for i in pred_classes]
    #     return bbox, labels

    def detect_object(self, batch_input):
        # img_list = [
        #     {'image': np.array(img)} for img in batch_input
        # ]
        if batch_input.dtype == torch.uint8:
            outputs = [self.detector(np.array(img).transpose(1, 2, 0)[..., ::-1]) for img in batch_input]
        else:
            outputs = [self.detector(np.array(img*255).astype(np.uint8).transpose(1, 2, 0)[..., ::-1]) for img in batch_input]
        pred_classes = [output["instances"].pred_classes for output in outputs]
        bbox = [output["instances"].pred_boxes for output in outputs]
        labels = [[self.total_classes[i] for i in pred] for pred in pred_classes]
        return bbox, labels

    def get_depth(self,):
        pass

    # def forward(self, raw_image):
    #     raw_image = raw_image.resize((self.H, self.W))
    #     caption = self.get_caption(raw_image)
    #     bbox, labels = self.detect_object(raw_image)
    #     return caption, bbox, labels
    
    def forward(self, batch_input):
        # print(batch_input.shape)
        # print(batch_input.shape)
        caption = self.get_caption(batch_input)
        bbox, labels = self.detect_object(batch_input)
        return caption, bbox, labels

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    VFM = VisualFoundationModels(device)
    import os
    import time
    from torchvision.transforms.functional import InterpolationMode

    blip_vis_processors = VFM.blip_vis_processors
    trans = transforms.Compose(
        [
            # transforms.Resize(
            #     (364, 364), interpolation=InterpolationMode.BICUBIC
            # ),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ]
    )
    tensors = []
    start_time = time.time()
    for file in os.listdir('tools/img_1'):
        img = os.path.join('tools/img_1', file)
        raw_image = Image.open(img).convert("RGB")
        # image_tensor = blip_vis_processors["eval"](raw_image).unsqueeze(0)
        image_tensor = trans(raw_image.resize((256, 256))).unsqueeze(0)
        # image_tensor = blip_vis_processors["eval"](image_tensor)
        # caption, bbox, labels = VFM(raw_image)
        print(f'----------------{file}---------------')
        # print(f'caption: {caption}')
        # print(f'bbox: {bbox}')
        # print(f'labels: {labels}')
        tensors.append(image_tensor)
    batch = torch.cat(tensors, dim=0)
    batch = (batch * 255).byte() # tensor.nint8
    caption, bbox, labels = VFM(batch)
    print(f'caption: {caption}')
    print(f'bbox: {bbox}')
    print(f'labels: {labels}')
    end_time = time.time()
    print(f'time: {round(end_time-start_time, 2)} s')
