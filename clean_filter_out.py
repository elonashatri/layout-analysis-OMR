import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_staff_lines_second_method(image_path, output_path='detected_staff_lines_second_method.png'):
    def is_staff_line(contour):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        return aspect_ratio > 100 and h < 20

    def split_contour(contour):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100:
            segments = []
            num_segments = max(1, w // 100)
            for i in range(num_segments):
                start = i * 100
                end = min((i + 1) * 100, w)
                segment_points = contour[:, :, 0][start:end]
                if len(segment_points) > 0:
                    segment = np.hstack((segment_points, contour[:, :, 1][start:end])).reshape(-1, 1, 2)
                    segments.append(segment)
            return segments
        else:
            return [contour]

    # Load the image
    image = cv2.imread(image_path)
    result = image.copy()

    # Convert to grayscale and apply Otsu's threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    staff_lines = []
    non_staff_lines = []

    # Process each contour and classify as staff line or not
    for c in cnts:
        for segment in split_contour(c):
            x, y, w, h = cv2.boundingRect(segment)
            if is_staff_line(segment):
                cv2.drawContours(result, [segment], -1, (0, 255, 0), 2)  # Green for staff lines
                staff_lines.append((x, y, w, h))
            else:
                cv2.drawContours(result, [segment], -1, (0, 0, 255), 2)  # Red for non-staff lines
                non_staff_lines.append((x, y, w, h))

    # Sort staff lines by y-coordinate (top to bottom)
    staff_lines = sorted(staff_lines, key=lambda line: line[1])
    
    # Print coordinates and classifications
    print("Detected Lines:")
    staff_id = 1
    for (x, y, w, h) in staff_lines:
        print(f"Staff Line {staff_id}: x={x}, y={y}, w={w}, h={h}")
        staff_id += 1
    for (x, y, w, h) in non_staff_lines:
        print(f"Non-Staff Line: x={x}, y={y}, w={w}, h={h}")

    # Save the result
    cv2.imwrite(output_path, result)

    # Display the result
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    original_image = cv2.imread(image_path)
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    result_image = cv2.imread(output_path)
    axes[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Detected Lines (Second Approach)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
image_path = '/Users/elona/Documents/layout-analysis-OMR/output_elastic_transform.png'
detect_staff_lines_second_method(image_path, 'elastic_output_second_method.png')
