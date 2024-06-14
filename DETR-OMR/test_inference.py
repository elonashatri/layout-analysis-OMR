import torch
import torchvision
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from coco_eval import CocoEvaluator
import timm  # Ensure timm is installed

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "custom_test.json")  # Ensure this file exists
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

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

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

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

# Load the model
checkpoint_path = '/Users/elona/Documents/layout-analysis-OMR/lightning_logs/version_4/checkpoints/epoch=16-step=10000.ckpt'
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False)
model.eval()

# Setup the processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Setup the test dataset and dataloader
test_dataset = CocoDetection(img_folder='/Users/elona/Documents/layout-analysis-OMR/data/test', processor=processor)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=2)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Setup COCO evaluator
evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=["bbox"])

# Inference and evaluation
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

coco_dt = []

for idx, batch in enumerate(tqdm(test_dataloader)):
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
    
    # Debug: Print loaded labels
    print(f"Batch {idx} labels: {labels}")
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0.1)
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    
    # Debug: Print predictions
    print(f"Batch {idx} predictions: {predictions}")
    
    formatted_predictions = prepare_for_coco_detection(predictions)
    
    # Debug: Print formatted predictions
    print(f"Batch {idx} formatted predictions: {formatted_predictions}")
    
    coco_dt.extend(formatted_predictions)
    evaluator.update(formatted_predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()

# Save evaluation results
eval_results = evaluator.coco_eval['bbox'].__dict__

# Serialize only relevant evaluation results to JSON
with open('test_evaluation_results.json', 'w') as f:
    json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in eval_results.items()}, f)

# Visualization for a random sample
pixel_values, target = test_dataset[np.random.randint(0, len(test_dataset))]
pixel_values = pixel_values.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values, pixel_mask=None)

image_id = target['image_id'].item()
image = test_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('/Users/elona/Documents/layout-analysis-OMR/data/test', image['file_name']))
width, height = image.size
postprocessed_outputs = processor.post_process_object_detection(outputs, target_sizes=[(height, width)], threshold=0.1)
results = postprocessed_outputs[0]
plot_results(image, results['scores'], results['labels'], results['boxes'])
