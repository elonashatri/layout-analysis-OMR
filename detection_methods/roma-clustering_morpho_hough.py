"""
This script detects staff lines in a music score image using clustering. It applies morphological operations and Hough line transform to detect lines, and then clusters them to filter out noise.

Parameters:
- image_path: Path to the input image
- output_path: Path to save the output image with detected lines
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os

def detect_staff_lines_clustering(image_path, output_path='roma_detected_staff_lines_colored.png'):
    def cluster_and_filter_lines(lines, eps=20, min_samples=5):
        y_coords = [line[0][1] for line in lines] + [line[0][3] for line in lines]
        y_coords = np.array(y_coords).reshape(-1, 1)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coords)
        labels = db.labels_
        unique_labels = set(labels)
        clustered_lines = []
        for label in unique_labels:
            if label == -1:
                continue
            label_indices = np.where(labels == label)[0]
            label_lines = [lines[i // 2] for i in label_indices]
            label_lines.sort(key=lambda line: line[0][1])
            mean_y = np.mean([line[0][1] for line in label_lines])
            filtered_label_lines = [line for line in label_lines if abs(line[0][1] - mean_y) < eps]
            if len(filtered_label_lines) >= min_samples:
                clustered_lines.append(filtered_label_lines)
        return clustered_lines

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    detected_lines = cv2.erode(detected_lines, None, iterations=1)
    detected_lines = cv2.dilate(detected_lines, None, iterations=1)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(detected_lines, 1, np.pi / 180, threshold=100, minLineLength=1900, maxLineGap=10)
    if lines is not None:
        clustered_staff_lines = cluster_and_filter_lines(lines)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        color_idx = 0
        for cluster in clustered_staff_lines:
            color = colors[color_idx % len(colors)]
            color_idx += 1
            for line in cluster:
                x1, y1, x2, y2 = line[0]
                cv2.line(color_image, (x1, y1), (x2, y2), color, 2)
    cv2.imwrite(output_path, color_image)



# Process all images in the input directory
input_dir = '/Users/elona/Documents/layout-analysis-OMR/augumented_images'
output_dir = '/Users/elona/Documents/layout-analysis-OMR/detected_lines_results'

for filename in os.listdir(input_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"roma_{filename}")
        detect_staff_lines_clustering(input_path, output_path)

