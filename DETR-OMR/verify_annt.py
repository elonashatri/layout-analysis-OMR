import os
import json
import torch
import torchvision
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import DataLoader
from tqdm import tqdm
from coco_eval import CocoEvaluator
from pycocotools.coco import COCO

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor):
        ann_file = os.path.join(img_folder, "custom_test.json")
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

# Paths
img_folder = '/Users/elona/Documents/layout-analysis-OMR/data/test'
ann_file = os.path.join(img_folder, "custom_test.json")

# Verify the structure of JSON file
with open(ann_file, 'r') as f:
    data = json.load(f)
    print(data.keys())
    print(data['annotations'][0])

# Initialize Processor and Dataset
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
test_dataset = CocoDetection(img_folder=img_folder, processor=processor)

# Check dataset length and sample item
print("Number of test examples:", len(test_dataset))
pixel_values, target = test_dataset[0]
print("Sample pixel values shape:", pixel_values.shape)
print("Sample target:", target)

# Create DataLoader
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=2)

# Initialize Model and Evaluator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
model.eval()
evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=["bbox"])

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

# Inference and Evaluation Loop
for idx, batch in enumerate(tqdm(test_dataloader)):
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0.5)
    
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    formatted_predictions = prepare_for_coco_detection(predictions)
    
    print(f"Batch {idx} formatted predictions: {formatted_predictions}")
    
    evaluator.update(formatted_predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()

# Save evaluation results
eval_results = evaluator.coco_eval['bbox'].__dict__
with open('evaluation_results.json', 'w') as f:
    json.dump(eval_results, f)
