from typing import Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import cv2
import torchvision
from torchvision import transforms
from torchvision import ops
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.resnet import ResNet50_Weights, resnet50
from transformers import DetrImageProcessor, DeformableDetrForObjectDetection

from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import shutil
import sys

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

image_folder = sys.argv[1]
part = int(sys.argv[2])
model_path = sys.argv[3]
output_folder = sys.argv[4]

if part == 1:
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

    def draw_boxes(boxes, labels, classes, image, color=None, text_pos=0):
        for i, box in enumerate(boxes):
            pos = {0 : (int(box[0]), int(box[1] - 5)),
                1 : (int(box[0]+5), int(box[3]-5))}
            pos_coord = pos[text_pos]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2
            )
            cv2.putText(image, classes[i], pos_coord,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        lineType=cv2.LINE_AA)
        return image

    coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
                'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'mal', 
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
                'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
                'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']


    # This will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

    def predict(input_tensor, model, device, detection_threshold):
        outputs = model(input_tensor)
        pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_labels = outputs[0]['labels'].cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        
        boxes, classes, labels, indices = [], [], [], []
        for index in range(len(pred_scores)):
            if pred_scores[index] >= detection_threshold:
                boxes.append(pred_bboxes[index].astype(np.int32))
                classes.append(pred_classes[index])
                labels.append(pred_labels[index])
                indices.append(index)
        boxes = np.int32(boxes)
        return boxes, classes, labels, indices
    
    batch_size = 1

    target_layers = [model.backbone]

    cam = AblationCAM(model,
                    target_layers,
                    reshape_transform=fasterrcnn_reshape_transform,
                    ablation_layer=AblationLayerFasterRCNN(),
                    ratio_channels_to_ablate=0.25)

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    with torch.no_grad():
        model.eval()
        for batch in range(len(images)//batch_size):
            print(batch, end='\r')
            input_tensor = [image.to(device) for image in images[batch*batch_size:(batch+1)*batch_size]]
            boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
            img_name = image_names[batch]
            
            input_tensor = torch.stack(input_tensor)
            targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
            grayscale_cam = cam(input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(input_tensor[0].permute(1,2,0).cpu().numpy(), grayscale_cam, use_rgb=True)
            image_with_bounding_boxes = draw_boxes(boxes, [12]*len(boxes), ['mal']*len(boxes), cam_image, color=(255,0,0), text_pos=1)
            Image.fromarray(image_with_bounding_boxes)
            rgb_image = cv2.cvtColor(image_with_bounding_boxes, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(output_folder, img_name), rgb_image)
            del cam_image
            del image_with_bounding_boxes
            for image in input_tensor:
                del image
            del input_tensor
            torch.cuda.empty_cache()
#         if batch >= 20:
#             break

elif part == 2:
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
    
    def draw_boxes(boxes, labels, classes, image, color=(255,0,0), text_pos=0):
        for i, box in enumerate(boxes):
            pos = {0 : (int(box[0]), int(box[1] - 5)),
                1 : (int(box[0]+5), int(box[3]-5))}
            pos_coord = pos[text_pos]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2
            )
            cv2.putText(image, classes[i], pos_coord,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        lineType=cv2.LINE_AA)
        return image

    coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
                'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'mal', 
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
                'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
                'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']


    # This will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

    def predict(pixel_values, pixel_mask, model, device, detection_threshold):
        outputs = model(pixel_values, pixel_mask)
        logits = softmax(outputs.logits.cpu().numpy()[0])
        pred_labels = np.argmax(logits, axis=1)
        pred_scores = np.max(logits, axis=1)
        pred_boxes = outputs.pred_boxes.cpu()
        pred_boxes = ops.box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        
        pred_classes = ['pred']*pred_labels.shape[0]
        
        boxes, classes, labels, indices, scores = [], [], [], [], []
        for index in range(pred_scores.shape[0]):
            if pred_scores[index] >= detection_threshold:
                boxes.append(pred_boxes[0][index].numpy())
                classes.append(pred_classes[index])
                labels.append(pred_labels[index])
                indices.append(index)
                scores.append(pred_scores[index])
        boxes = np.array(boxes)
        return boxes, classes, labels, indices, scores
    
    class HuggingfaceToTensorModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(HuggingfaceToTensorModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            return self.model(x).logits

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    target_layers = [model.model.backbone.conv_encoder.model.layer4[2]]
    cam = EigenCAM(model=HuggingfaceToTensorModelWrapper(model),
                target_layers=target_layers,
                reshape_transform=None)



    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(test_dataloader):
            print('batch:',i)
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            for j in range(len(batch['pixel_values'])):
                input_tensor = pixel_values[j].unsqueeze(0)
                boxes, classes, labels, indices, scores = predict(pixel_values[j].unsqueeze(0), pixel_mask[j].unsqueeze(0), model, device, 0.875)
                grayscale_cam = cam(input_tensor, targets=None)
                cam_image = show_cam_on_image(transforms.ToTensor()(transforms.ToPILImage()(input_tensor[0])).permute(1,2,0).cpu().numpy(),
                                                                    grayscale_cam[0], use_rgb=True)
                image_with_bounding_boxes = cam_image.copy()
                image_height, image_width = batch['labels'][j]['size'][0].item(), batch['labels'][j]['size'][1].item()
                image_id = batch['labels'][j]['image_id'].item()
                img_name = test_dataset.coco.loadImgs(image_id)[0]['file_name']
                img_path = os.path.join(image_folder, img_name)
                # img_with_labels = cv2.imread(img_path)
                if boxes.shape[0]:
                    keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0)
                    boxes = torch.tensor(boxes[keep])
                    if len(boxes.shape) == 1:
                        boxes = boxes.unsqueeze(0)
                    boxes[:,0] *= image_width
                    boxes[:,1] *= image_height
                    boxes[:,2] *= image_width
                    boxes[:,3] *= image_height
                    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image, color=(0,255,0), text_pos=1)
                rgb_image = cv2.cvtColor(image_with_bounding_boxes, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(output_folder, img_name), rgb_image)
#                 plt.imshow(image_with_bounding_boxes)
#                 plt.show()
            del pixel_values, pixel_mask
            torch.cuda.empty_cache()
            
    #         if i > 10:
    #             break