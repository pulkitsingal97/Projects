from typing import Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import json
import cv2
import torchvision
from torchvision import transforms
from torchvision import ops
from torchvision.ops import misc as misc_nn_ops
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.resnet import ResNet50_Weights, resnet50
import sys
import shutil
from transformers import DetrImageProcessor, DeformableDetrForObjectDetection

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

image_folder = sys.argv[1]
part = int(sys.argv[2])
model_path = sys.argv[3]
preds_folder = sys.argv[4]


if part == 1:
    # Reading images
    def readImages(data_dir):
        images = []
        image_names = []
        for image in os.listdir(data_dir):
            image_names.append(image)
            img_path = os.path.join(data_dir, image)
            img = cv2.imread(img_path)
            img = transforms.ToTensor()(img)
            images.append(img)
        return images, image_names

    images, image_names = readImages(image_folder)

    # Model definition
    def fasterrcnn_resnet50_fpn_self(
        *,
        weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers: Optional[int] = None,
        box_nms_thresh=0.5,
        **kwargs: Any,) -> FasterRCNN:
        
        weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
        weights_backbone = ResNet50_Weights.verify(weights_backbone)

        if weights is not None:
            weights_backbone = None
            num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        elif num_classes is None:
            num_classes = 91

        is_trained = weights is not None or weights_backbone is not None
        trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
        norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

        backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
        model = FasterRCNN(backbone, num_classes=num_classes, box_nms_thresh=box_nms_thresh, **kwargs)

        if weights is not None:
            model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
            if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
                overwrite_eps(model, 0.0)

        return model


    model = fasterrcnn_resnet50_fpn_self(box_nms_thresh=0).to(device)

    # Loading model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Inference
    batch_size = 1
    if os.path.exists(preds_folder):
        shutil.rmtree(preds_folder)
    os.makedirs(preds_folder)

    with torch.no_grad():
        model.eval()
        for batch in range(len(images)//batch_size):
            print(batch, end='\r')
            test_images = [image.to(device) for image in images[batch*batch_size:(batch+1)*batch_size]]
            output = model(test_images)
            for i in range(batch_size):
                img_name = image_names[batch*batch_size+i]
                pred_file = img_name[:-4] + '_preds.txt'
                boxes = ops.box_convert(output[i]['boxes'], in_fmt='xyxy', out_fmt='cxcywh').cpu().numpy()
                img_height, img_width = test_images[i].shape[1:3]
                boxes[:,0] /= img_width
                boxes[:,2] /= img_width
                boxes[:,1] /= img_height
                boxes[:,3] /= img_height
                scores = output[i]['scores'].cpu().numpy().reshape(-1, 1)
                op = np.concatenate((boxes, scores), axis=1)
                
                np.savetxt(os.path.join(preds_folder, pred_file), op, fmt='%f')
            for image in test_images:
                del image
            del test_images
            del output
            torch.cuda.empty_cache()

elif part == 2:

    # Creating test json file
    dict_main = {}
    dict_main['categories'] = [
                                    {
                                        "id": 0,
                                        "name": "mal",
                                        "supercategory": 'null'
                                    }
                                ]
    dict_main['images'] = []

    for i, image in enumerate(os.listdir(image_folder)):
        img = cv2.imread(os.path.join(image_folder, image))
        img_height, img_width = img.shape[:2]
        dict_main['images'].append({'file_name': image,
                                'height': img_height,
                                    'width': img_width,
                                    'id': i,
                                    'depth': 3})
    dict_main['annotations'] = []

    output_json_file = 'instances_test.json'
    with open(output_json_file, "w") as json_file:
        json.dump(dict_main, json_file, indent=4)


    # Creating cocodetection dataset
    class CocoDetection(torchvision.datasets.CocoDetection):
        def __init__(self, img_folder, json_path, processor, train=True):
            ann_file = json_path
            super(CocoDetection, self).__init__(img_folder, ann_file)
            self.processor = processor

        def __getitem__(self, idx):
            img, target = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            encoding = self.processor(images=img, annotations=target, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()
            target = encoding["labels"][0]

            return pixel_values, target
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    test_dataset = CocoDetection(img_folder=image_folder, json_path=output_json_file, processor=processor, train=False)


    # Dataloader
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
    
    batch_size=4
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=3)

    # Model
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").to(device)
    model.load_state_dict(torch.load(model_path))


    def softmax(x):
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        softmax_x = exp_x / sum_exp_x
        return softmax_x

    # Inference
    batch_count = 0
    score_thres = 0.8
    
    if os.path.exists(preds_folder):
        shutil.rmtree(preds_folder)
    os.makedirs(preds_folder)
    with torch.no_grad():
        model.eval()
        for batch in test_dataloader:
            batch_count += 1
            print(batch_count, end='\r')
            outputs = model(batch['pixel_values'].to(device), batch['pixel_mask'].to(device))
            for i in range(len(batch['labels'])):
                tumor_pred = False
                image_id = batch['labels'][i]['image_id'].item()
                image = test_dataset.coco.loadImgs(image_id)[0]
                image_name = image['file_name']
                pred_file = image_name[:-4] + '_preds.txt'
                scores = softmax(outputs.logits[i].cpu().numpy())
                scores = scores[:, 0]
                idx = np.where(scores > score_thres)
                scores = scores[idx].reshape(-1, 1)
                pred_boxes = outputs.pred_boxes[i].cpu().numpy()
                pred_boxes = pred_boxes[idx]
                op = np.concatenate((pred_boxes, scores), axis=1)
                np.savetxt(os.path.join(preds_folder, pred_file), op, fmt='%f')

else:
    print('Error! Acceptable values for part(2nd argument) = {1, 2}')