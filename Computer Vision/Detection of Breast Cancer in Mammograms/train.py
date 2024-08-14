from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import cv2
import torchvision
from torchvision import transforms
from torchvision.ops import misc as misc_nn_ops
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import time
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.resnet import ResNet50_Weights, resnet50
import sys
from transformers import DetrImageProcessor, DeformableDetrForObjectDetection, TrainingArguments, DetrFeatureExtractor, Trainer
import wandb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_dir = sys.argv[1]
part = int(sys.argv[2])

if part == 1:
    # Reading data
    def readYOLOData(data_dir):
        images = []
        targets = []
        annot_files = [file_name for file_name in sorted(os.listdir(os.path.join(data_dir, 'labels')))]
        
        for annot_file in annot_files:
            image_name = annot_file[:-4] + '.png'
            image_path = os.path.join(data_dir, 'images', image_name)
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
            img = transforms.ToTensor()(img)
            images.append(img)
            
            boxes_txt = []
            boxes = []            
            annot_path = os.path.join(data_dir, 'labels', annot_file)
            with open(annot_path, 'r') as file:
                for line in file:
                    values = line.strip().split(' ')
                    boxes_txt.append([float(val) for val in values[1:]])
            labels = [12]*len(boxes_txt)
            for i, box in enumerate(boxes_txt):
                x_min = (box[0]-box[2]/2)*img_width
                x_max = (box[0]+box[2]/2)*img_width
                y_min = (box[1]-box[3]/2)*img_height
                y_max = (box[1]+box[3]/2)*img_height
                boxes.append([x_min, y_min, x_max, y_max])
            targets.append({'boxes':torch.tensor(boxes), 'labels': torch.tensor(labels)})
        
        return images, targets

    train_mal_images, train_targets = readYOLOData(os.path.join(data_dir, 'yolo_1k', 'train'))
    val_mal_images, val_targets = readYOLOData(os.path.join(data_dir, 'yolo_1k', 'val'))

    # Model
    def fasterrcnn_resnet50_fpn_self(
        *,
        weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers: Optional[int] = None,
        box_nms_thresh=0.5,
        **kwargs: Any,
    ) -> FasterRCNN:
        
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


    # Training
    num_epochs = 80
    lr = 0.001
    batch_size = 8
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


    train_loss = []
    val_loss = []
    num_train_batches = len(train_mal_images)//batch_size
    num_val_batches = len(val_mal_images)//batch_size

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        total_train_loss = 0
        for i in range(num_train_batches):
            images = train_mal_images[i*batch_size:(i+1)*batch_size]
            images = [image.to(device) for image in images]
            targets = train_targets[i*batch_size:(i+1)*batch_size]
            targets = [{'boxes' : targets[i]['boxes'].to(device), 'labels': targets[i]['labels'].to(device)} for i in range(batch_size)]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_train_loss += losses
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            for image in images:
                del image
            for target in targets:
                for val in target.values():
                    del val
            for val in loss_dict.values():
                del val
            torch.cuda.empty_cache()
            
    #     if (epoch+1)%5 == 0:
        
        train_loss.append(total_train_loss/(num_train_batches*batch_size))
        
        with torch.no_grad():
    #         model.eval()
            total_val_loss = 0
            for batch in range(num_val_batches):
                images = [image.to(device) for image in val_mal_images[batch*batch_size:(batch+1)*batch_size]]
                targets = val_targets[batch*batch_size:(batch+1)*batch_size]
                targets = [{'boxes' : targets[i]['boxes'].to(device), 'labels': targets[i]['labels'].to(device)} for i in range(batch_size)]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses
                
                for image in images:
                    del image
                for target in targets:
                    for val in target.values():
                        del val
                for val in loss_dict.values():
                    del val
                torch.cuda.empty_cache()
            
            val_loss.append(total_val_loss/(num_val_batches*batch_size))
        end_time = time.time()
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")
        print(f'Time: {(end_time-start_time)/60:.4f}')
        print()
    
    torch.save(model.state_dict(), 'model_1.pth')

elif part == 2:
    wandb.init(mode="disabled")
    class CocoDetection(torchvision.datasets.CocoDetection):
        def __init__(self, img_folder, json_path, processor, train=True):
            ann_file = json_path
            super(CocoDetection, self).__init__(img_folder, ann_file)
            self.processor = processor

        def __getitem__(self, idx):
            # read in PIL image and target in COCO format
            # feel free to add data augmentation here before passing them to the next step
            img, target = super(CocoDetection, self).__getitem__(idx)

            # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            encoding = self.processor(images=img, annotations=target, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
            target = encoding["labels"][0] # remove batch dimension

            return pixel_values, target
        
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = CocoDetection(img_folder=os.path.join(data_dir, 'coco_1k', 'train2017'),
                                  json_path=os.path.join(data_dir, 'coco_1k', 'annotations', 'instances_train2017.json'),
                                  processor=processor)
    val_dataset = CocoDetection(img_folder=os.path.join(data_dir, 'coco_1k', 'val2017'),
                                json_path=os.path.join(data_dir, 'coco_1k', 'annotations', 'instances_train2017.json'),
                                processor=processor,
                                train=False)
    
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
    
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=3, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=3)

    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").to(device)

    training_args = TrainingArguments(
        output_dir="/kaggle/working/results",
        per_device_train_batch_size=4,
        num_train_epochs=30,
        fp16=False,
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    feature_extractor = DetrFeatureExtractor.from_pretrained("SenseTime/deformable-detr")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
    )

    trainer.train()

    torch.save(model.state_dict(), 'model_1.pth')

else:
    print('Error! Acceptable values of part(2nd argument) = {1, 2}')