"""
This script detects staff lines in a music score image using morphological operations and contour detection. It applies thresholding, morphological operations, and contour drawing to detect lines.

Parameters:
- image_path: Path to the input image
- output_path: Path to save the output image with detected lines
"""

import cv2
import os
import numpy as np

def detect_staff_lines_second_method(image_path, output_path='florence_detected_staff_lines_second_method.png'):
    image = cv2.imread(image_path)
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (36, 255, 12), 2)
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    # detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (36, 255, 12), 2)
    cv2.imwrite(output_path, result)


# Process all images in the input directory
input_dir = '/Users/elona/Documents/layout-analysis-OMR/new_transform'
output_dir = '/Users/elona/Documents/layout-analysis-OMR/detected_lines_results'

for filename in os.listdir(input_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"new_transform_florence_{filename}")
        detect_staff_lines_second_method(input_path, output_path)