import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter
import random

def add_gaussian_noise(image, mean=0, var=10):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss.reshape(row, col, ch)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = int(salt_prob * total_pixels)
    num_pepper = int(pepper_prob * total_pixels)
    
    # Salt noise
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255
    
    # Pepper noise
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy

def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted_image = np.zeros_like(image)
    for i in range(shape[2]):
        distorted_image[:, :, i] = map_coordinates(image[:, :, i], indices, order=1, mode='reflect').reshape(shape[:2])

    return distorted_image

def perspective_transform(image):
    rows, cols, ch = image.shape
    pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
    pts2 = pts1 + np.float32([[random.uniform(-50, 50), random.uniform(-50, 50)] for _ in range(4)])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (cols, rows))

def blur_image(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Example usage of augmentation methods
image_path = '/Users/elona/Documents/layout-analysis-OMR/01 - Full score - Vivaldi_-_Violin_Concerto_in_F_minor_Op._8_No._4_RV._297_Winter_for_Solo_Piano-v2 - 008.png'
original_image = cv2.imread(image_path)

# Apply augmentations
gaussian_noisy_image = add_gaussian_noise(original_image)
salt_pepper_noisy_image = add_salt_and_pepper_noise(original_image)
elastic_transformed_image = elastic_transform(original_image, alpha=34, sigma=4)
perspective_transformed_image = perspective_transform(original_image)
blurred_image = blur_image(original_image)

# Save the augmented images
cv2.imwrite('output_gaussian_noise.png', gaussian_noisy_image)
cv2.imwrite('output_salt_pepper_noise.png', salt_pepper_noisy_image)
cv2.imwrite('output_elastic_transform.png', elastic_transformed_image)
cv2.imwrite('output_perspective_transform.png', perspective_transformed_image)
cv2.imwrite('output_blurred.png', blurred_image)

# Display the results
fig, axes = plt.subplots(2, 3, figsize=(20, 15))
axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(gaussian_noisy_image, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Gaussian Noise')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(salt_pepper_noisy_image, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Salt-and-Pepper Noise')
axes[0, 2].axis('off')

axes[1, 0].imshow(cv2.cvtColor(elastic_transformed_image, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Elastic Transformation')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(perspective_transformed_image, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Perspective Transformation')
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Blurred Image')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
