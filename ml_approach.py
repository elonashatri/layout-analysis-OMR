import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

def load_image_and_annotations(image_path, xml_path):
    image = cv2.imread(image_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for node in root.findall('.//Node'):
        if node.get('ClassName') == 'kStaffLine':
            top = int(node.find('Top').text)
            left = int(node.find('Left').text)
            width = int(node.find('Width').text)
            height = int(node.find('Height').text)
            annotations.append((left, top, width, height))
    return image, annotations

def apply_perspective_transform(image, annotations):
    rows, cols, ch = image.shape
    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    pts2 = pts1 + np.float32([[np.random.uniform(-50, 50), np.random.uniform(-50, 50)] for _ in range(4)])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_image = cv2.warpPerspective(image, M, (cols, rows))
    
    transformed_annotations = []
    for left, top, width, height in annotations:
        corners = np.float32([
            [left, top],
            [left + width, top],
            [left, top + height],
            [left + width, top + height]
        ])
        transformed_corners = cv2.perspectiveTransform(np.array([corners]), M)[0]
        x_min = min(transformed_corners[:, 0])
        y_min = min(transformed_corners[:, 1])
        x_max = max(transformed_corners[:, 0])
        y_max = max(transformed_corners[:, 1])
        transformed_annotations.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))
    
    return transformed_image, transformed_annotations

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def preprocess_data(image, annotations):
    processed_images = []
    labels = []
    for left, top, width, height in annotations:
        # Crop and resize each staff line region to a fixed size
        staff_line = image[top:top + height, left:left + width]
        staff_line = cv2.resize(staff_line, (128, 128))
        staff_line = staff_line.reshape((128, 128, 1)).astype('float32') / 255
        processed_images.append(staff_line)
        labels.append(1)  # Label for staff line regions

    # Add negative examples (non-staff line regions)
    for _ in range(len(annotations)):
        h, w = image.shape[:2]
        x = np.random.randint(0, w - 128)
        y = np.random.randint(0, h - 128)
        non_staff_line = image[y:y + 128, x:x + 128]
        if non_staff_line.shape == (128, 128, 1):
            processed_images.append(non_staff_line)
            labels.append(0)  # Label for non-staff line regions

    return np.array(processed_images), np.array(labels)

# Example usage
image_path = '/path/to/your/image.png'
xml_path = '/path/to/your/annotations.xml'
image, annotations = load_image_and_annotations(image_path, xml_path)

# Apply augmentation
transformed_image, transformed_annotations = apply_perspective_transform(image, annotations)

# Prepare data for training
X_train, y_train = preprocess_data(transformed_image, transformed_annotations)

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save('staff_line_detection_model.h5')

# Inference function to detect staff lines in a new image
def detect_staff_lines_in_image(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image.reshape((1, 128, 128, 1)).astype('float32') / 255
    prediction = model.predict(image)
    return prediction

# Example inference
prediction = detect_staff_lines_in_image('/path/to/new/image.png', model)
print(f'Prediction: {prediction}')
