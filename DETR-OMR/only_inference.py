import os
import json
import torch
import torchvision
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pytorch_lightning as pl

# Initialize paths
img_folder = '/Users/elona/Documents/layout-analysis-OMR/data/test'
img_path = '/Users/elona/Documents/layout-analysis-OMR/data/test/1-2-Kirschner_-_Chissà_che_cosa_pensa-004.png'
ann_file = os.path.join(img_folder, "custom_test.json")
checkpoint_path = '/Users/elona/Documents/layout-analysis-OMR/lightning_logs/version_4/checkpoints/epoch=16-step=10000.ckpt'

# Load COCO dataset
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor):
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

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

# Initialize processor and dataset
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
test_dataset = CocoDetection(img_folder=img_folder, processor=processor)

# Create DataLoader
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=2)

# Define the Detr model class
class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm",
            num_labels=217,  # Adjust the number of labels as per your dataset
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

# Load model from checkpoint
model = Detr.load_from_checkpoint(checkpoint_path, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# Function to convert bounding boxes
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

# Function to prepare predictions for COCO format
def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        coco_results.extend(
            [
                {"image_id": original_id, "category_id": labels[k], "bbox": box, "score": scores[k]}
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

# Load and process the specified image
image = Image.open(img_path)
if image.mode != 'RGB':
    image = image.convert('RGB')

# Perform inference on the specified image
encoding = processor(images=image, return_tensors="pt")
pixel_values = encoding["pixel_values"].to(model.device)
pixel_mask = encoding["pixel_mask"].to(model.device)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

# Post-process the outputs
target_sizes = torch.tensor([image.size[::-1]], device=model.device)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

# Plot the results on the image
def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], 
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        text = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

plot_results(image, results['scores'], results['labels'], results['boxes'])
