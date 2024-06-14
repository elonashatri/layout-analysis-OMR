import os
import datetime
import glob
import shutil
from tqdm import tqdm
from xml.dom import minidom
from PIL import Image
import json

ANNOTATIONS_PATH = "/Users/elona/Desktop/merged_doremi/doremi_final_june24/xml_by_page/*.xml"
IMAGES_PATH = "/Users/elona/Desktop/merged_doremi/doremi_final_june24/Images"
CATEGORIES_MAPPING_PATH = "/Users/elona/Documents/layout-analysis-OMR/mapping.json"
OUTPUT_PATH = "test_annot"
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.9
MAX_SCORES = 100

def prepare_coco_annotations():
    print("Preparing COCO-style annotations...")

    # Load category mappings
    ids_classnames = {}
    categories = []
    with open(CATEGORIES_MAPPING_PATH) as json_file:
        data = json.load(json_file)
        for item in data:
            ids_classnames[item["name"]] = item["id"]
            category = {
                "supercategory": "type",
                "id": item["id"],
                "name": item["name"]
            }
            categories.append(category)

    xml_files = glob.glob(ANNOTATIONS_PATH)[:MAX_SCORES * 3]  # Limit to 300 files to allow splitting
    total_count = len(xml_files)
    train_images, train_annotations = [], []
    test_images, test_annotations = [], []
    validate_images, validate_annotations = [], []

    current_annotation_id = 1
    for current_iteration, xml_file in enumerate(tqdm(xml_files, desc="XML Files")):
        filename = os.path.basename(xml_file)[:-4]  # Remove .xml from the end

        # Parse XML Document
        xmldoc = minidom.parse(xml_file)

        img_filename = filename
        img_path = os.path.join(IMAGES_PATH, f"{img_filename}.png")
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Image dictionary for COCO
        image_dict = {
            "id": current_iteration + 1,
            "width": img_width,
            "height": img_height,
            "file_name": f"{img_filename}.png",
            "license": 0,  # Index in the Licenses Array
            "flickr_url": f"{img_filename}.png",
            "coco_url": f"{img_filename}.png",
            "date_captured": ""
        }

        nodes = xmldoc.getElementsByTagName("Node")

        for node in nodes:
            node_classname_str = node.getElementsByTagName("ClassName")[0].firstChild.data
            node_top_int = int(node.getElementsByTagName("Top")[0].firstChild.data)
            node_left_int = int(node.getElementsByTagName("Left")[0].firstChild.data)
            node_width_int = int(node.getElementsByTagName("Width")[0].firstChild.data)
            node_height_int = int(node.getElementsByTagName("Height")[0].firstChild.data)
            node_mask_str = node.getElementsByTagName("Mask")[0].firstChild.data 

            # Fix widths and heights if necessary
            if node_width_int == 0:
                node_width_int = 2
                node_left_int -= 1
            if node_height_int == 0:
                node_height_int = 2
                node_top_int -= 1
            
            # Get category id from dict
            category_id = ids_classnames[node_classname_str]
            
            counts = []

            for bits_counts in node_mask_str.split(" "):
                if bits_counts == "":
                    continue
                bit_count = bits_counts.split(":")
                counts.append(bit_count[0])
            segmentation = {}
            segmentation["counts"] = counts
            segmentation["size"] = [node_height_int,node_width_int]


            annotation = {
                "id": current_annotation_id,
                "image_id": current_iteration + 1,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": node_width_int * node_height_int,
                "bbox": [node_left_int, node_top_int, node_width_int, node_height_int],
                "iscrowd": 1
            }
            current_annotation_id += 1

            if current_iteration < int(total_count * TRAIN_SPLIT):
                train_annotations.append(annotation)
            elif current_iteration < int(total_count * TEST_SPLIT):
                test_annotations.append(annotation)
            else:
                validate_annotations.append(annotation)

        if current_iteration < int(total_count * TRAIN_SPLIT):
            train_images.append(image_dict)
            dest_dir = os.path.join(OUTPUT_PATH, "train_images")
        elif current_iteration < int(total_count * TEST_SPLIT):
            test_images.append(image_dict)
            dest_dir = os.path.join(OUTPUT_PATH, "test_images")
        else:
            validate_images.append(image_dict)
            dest_dir = os.path.join(OUTPUT_PATH, "validate_images")

        # Copy image to the destination directory
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.copy(img_path, os.path.join(dest_dir, f"{img_filename}.png"))

    # Prepare info dictionary
    now = datetime.datetime.now()
    info = {
        "year": 2024,
        "version": 1.0,
        "description": "DOREMI Dataset for OMR",
        "contributor": "Elona Shatri",
        "url": "https://github.com/penestia/DOREMI",
        "date_created": now.strftime("%a %b %d %Y %H:%M:%S") + " GMT+0100 (British Summer Time)"
    }

    # License dictionary
    doremi_license = {
        "id": 0,
        "name": "Unknown License",
        "url": ""
    }

    def create_coco_dict(images, annotations):
        return {
            "info": info,
            "images": images,
            "annotations": annotations,
            "licenses": [doremi_license],
            "categories": categories
        }

    train_coco_dict = create_coco_dict(train_images, train_annotations)
    test_coco_dict = create_coco_dict(test_images, test_annotations)
    validate_coco_dict = create_coco_dict(validate_images, validate_annotations)

    with open(os.path.join(OUTPUT_PATH, "custom_train.json"), 'w') as fp:
        json.dump(train_coco_dict, fp)

    with open(os.path.join(OUTPUT_PATH, "custom_test.json"), 'w') as fp:
        json.dump(test_coco_dict, fp)

    with open(os.path.join(OUTPUT_PATH, "custom_val.json"), 'w') as fp:
        json.dump(validate_coco_dict, fp)

def main():
    prepare_coco_annotations()

if __name__ == "__main__":
    main()
