"""
This script contains functions to detect different types of transformations applied to an image. 
It uses this information to apply the appropriate staff line detection method.
"""

import cv2
import numpy as np
import os

def detect_perspective_transform(image):
    """
    Detects perspective transformation by analyzing skewness using horizontal projections.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    horizontal_projection = np.sum(gray, axis=1)
    
    # Check for skewness in the horizontal projection
    diffs = np.diff(horizontal_projection)
    std_diff = np.std(diffs)
    print(f"Perspective transform - Std of horizontal projection diffs: {std_diff}")
    
    return std_diff > 1000  # Threshold may need tuning

def detect_salt_and_pepper_noise(image):
    """
    Detects salt and pepper noise by checking for random spikes in pixel intensity along horizontal lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    horizontal_projection = np.sum(gray, axis=1)
    
    # Check for random spikes
    spikes = np.sum(horizontal_projection > np.mean(horizontal_projection) + 3 * np.std(horizontal_projection))
    print(f"Salt and pepper noise - Number of spikes: {spikes}")
    
    return spikes > 10  # Threshold may need tuning

def detect_elastic_transform(image):
    """
    Detects elastic transformations by analyzing distortions in horizontal projection profiles.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    horizontal_projection = np.sum(gray, axis=1)
    
    # Check for distortions
    diffs = np.diff(horizontal_projection)
    std_diff = np.std(diffs)
    print(f"Elastic transform - Std of horizontal projection diffs: {std_diff}")
    
    return 500 < std_diff < 1000  # Threshold may need tuning

def detect_transformation(image_path):
    """
    Detects the type of transformation applied to the image.
    """
    image = cv2.imread(image_path)
    if detect_perspective_transform(image):
        return 'perspective'
    elif detect_salt_and_pepper_noise(image):
        return 'salt_and_pepper'
    elif detect_elastic_transform(image):
        return 'elastic'
    else:
        return 'other'


def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            input_path = os.path.join(input_dir, filename)
            transformation_type = detect_transformation(input_path)
            print(f"{filename}: Detected transformation - {transformation_type}")
            # output_path = os.path.join(output_dir, f"{transformation_type}_{filename}")
            


# Example usage
input_dir = '/Users/elona/Documents/layout-analysis-OMR/augumented_images'
output_dir = '/Users/elona/Documents/layout-analysis-OMR/detected_lines_results'
process_images(input_dir, output_dir)