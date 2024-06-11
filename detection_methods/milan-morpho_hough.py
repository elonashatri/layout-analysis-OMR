"""
This script detects staff lines in a music score image using morphological operations and Hough line transform. It applies thresholding, morphological operations, and Hough line transform to detect lines.

Parameters:
- image_path: Path to the input image
- output_path: Path to save the output image with detected lines
"""

import cv2
import numpy as np
import os

def detect_staff_lines_third_method(image_path, output_path='milan_detected_staff_lines_third_method.png'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    detected_lines = cv2.erode(detected_lines, None, iterations=1)
    detected_lines = cv2.dilate(detected_lines, None, iterations=1)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(detected_lines, 1, np.pi / 180, threshold=100, minLineLength=2000, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    detected_lines_color = cv2.cvtColor(detected_lines, cv2.COLOR_GRAY2BGR)
    combined_image = cv2.addWeighted(color_image, 0.7, detected_lines_color, 0.3, 0)
    cv2.imwrite(output_path, combined_image)



# Process all images in the input directory
input_dir = '/Users/elona/Documents/layout-analysis-OMR/augumented_images'
output_dir = '/Users/elona/Documents/layout-analysis-OMR/detected_lines_results'

for filename in os.listdir(input_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"milan_{filename}")
        detect_staff_lines_third_method(input_path, output_path)