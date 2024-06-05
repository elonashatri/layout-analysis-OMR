import numpy as np
import cv2
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def find_best_rotation_angle(image, angles):
    best_angle = 0
    max_variance = 0

    for angle in angles:
        rotated = rotate_image(image, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        hist = np.sum(thresh, axis=1)
        variance = np.var(hist)
        
        if variance > max_variance:
            max_variance = variance
            best_angle = angle
    
    return best_angle

def detect_staff_lines(image, output_path='detected_staff_lines.png'):
    def is_staff_line(x1, y1, x2, y2, min_length=200):
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        return length > min_length and (angle < 10 or angle > 170)

    # Find the best rotation angle
    angles = np.arange(-5, 5, 0.5)
    best_angle = find_best_rotation_angle(image, angles)
    print(f"Best rotation angle: {best_angle}")

    # Rotate the image to the best angle
    rotated_image = rotate_image(image, best_angle)

    # Convert to grayscale and apply Otsu's threshold
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect edges using Canny
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # Detect lines using Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    staff_lines = []
    non_staff_lines = []

    # Process each line and classify as staff line or not
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if is_staff_line(x1, y1, x2, y2):
                cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for staff lines
                staff_lines.append((x1, y1, x2, y2))
            else:
                cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for non-staff lines
                non_staff_lines.append((x1, y1, x2, y2))

    # Sort staff lines by y-coordinate (top to bottom)
    staff_lines = sorted(staff_lines, key=lambda line: min(line[1], line[3]))

    # Print coordinates and classifications
    print("Detected Lines:")
    staff_id = 1
    for (x1, y1, x2, y2) in staff_lines:
        print(f"Staff Line {staff_id}: ({x1}, {y1}) to ({x2}, {y2})")
        staff_id += 1
    for (x1, y1, x2, y2) in non_staff_lines:
        print(f"Non-Staff Line: ({x1}, {y1}) to ({x2}, {y2})")

    # Save the result
    cv2.imwrite(output_path, rotated_image)

    # Display the result
    fig, axes = plt.subplots(1, 3, figsize=(20, 15))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Corrected Image')
    axes[1].axis('off')

    result_image = cv2.imread(output_path)
    axes[2].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Detected Lines (Corrected)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
image_path = '/Users/elona/Documents/layout-analysis-OMR/outputs/output_perspective_transform.png'
original_image = cv2.imread(image_path)
if original_image is None:
    print(f"Error reading image from {image_path}. Please check the file path and try again.")
else:
    detect_staff_lines(original_image, 'staff_line_detection_with_rotation_and_inpainting_output.png')
