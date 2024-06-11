import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter
import random

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, var=10):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss.reshape(row, col, ch)
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Function to add salt and pepper noise to an image
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = int(salt_prob * total_pixels)
    num_pepper = int(pepper_prob * total_pixels)
    
    # Add salt noise
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255
    
    # Add pepper noise
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy

# Function to apply elastic transformation to an image
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

# Function to apply perspective transformation to an image
def perspective_transform(image):
    rows, cols, ch = image.shape
    pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
    pts2 = pts1 + np.float32([[random.uniform(-50, 50), random.uniform(-50, 50)] for _ in range(4)])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (cols, rows))

# Function to blur an image
def blur_image(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Function to simulate fading effect on an image
def fade_image(image, strength=0.5):
    faded = image.astype(np.float32) * strength
    return np.clip(faded, 0, 255).astype(np.uint8)

# Function to add vignetting effect to an image
def add_vignetting(image, strength=0.5):
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette = np.copy(image)
    
    for i in range(3):
        vignette[:,:,i] = vignette[:,:,i] * mask
    
    return cv2.addWeighted(image, 1 - strength, vignette.astype(np.uint8), strength, 0)

# Function to add a highlight to a specific region in an image
def add_highlight(image, x, y, width, height, color=(0, 255, 0), alpha=0.5):
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# Example usage of augmentation methods
image_path = '/Users/elona/Documents/layout-analysis-OMR/01 - Full score - Vivaldi_-_Violin_Concerto_in_F_minor_Op._8_No._4_RV._297_Winter_for_Solo_Piano-v2 - 008.png'
original_image = cv2.imread(image_path)

# Apply augmentations
gaussian_noisy_image = add_gaussian_noise(original_image)
salt_pepper_noisy_image = add_salt_and_pepper_noise(original_image)
elastic_transformed_image = elastic_transform(original_image, alpha=34, sigma=4)
perspective_transformed_image = perspective_transform(original_image)
blurred_image = blur_image(original_image)
faded_image = fade_image(original_image)
vignette_image = add_vignetting(original_image)
highlighted_image = add_highlight(original_image, 50, 50, 200, 100)

# Save the augmented images
cv2.imwrite('/Users/elona/Documents/layout-analysis-OMR/augumented_images/gaussian_noise.png', gaussian_noisy_image)
cv2.imwrite('/Users/elona/Documents/layout-analysis-OMR/augumented_images/salt_pepper_noise.png', salt_pepper_noisy_image)
cv2.imwrite('/Users/elona/Documents/layout-analysis-OMR/augumented_images/elastic_transform.png', elastic_transformed_image)
cv2.imwrite('/Users/elona/Documents/layout-analysis-OMR/augumented_images/perspective_transform.png', perspective_transformed_image)
cv2.imwrite('/Users/elona/Documents/layout-analysis-OMR/augumented_images/blurred.png', blurred_image)
cv2.imwrite('/Users/elona/Documents/layout-analysis-OMR/augumented_images/faded.png', faded_image)
cv2.imwrite('/Users/elona/Documents/layout-analysis-OMR/augumented_images/vignette.png', vignette_image)
cv2.imwrite('/Users/elona/Documents/layout-analysis-OMR/augumented_images/highlighted.png', highlighted_image)

# Display the results
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
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

axes[2, 0].imshow(cv2.cvtColor(faded_image, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title('Faded Image')
axes[2, 0].axis('off')

axes[2, 1].imshow(cv2.cvtColor(vignette_image, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title('Vignette Effect')
axes[2, 1].axis('off')

axes[2, 2].imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title('Highlighted Text')
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()
