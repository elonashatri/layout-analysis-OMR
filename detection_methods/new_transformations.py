import itertools
import numpy as np
from skimage import transform as stf
from numpy import random, floor
from PIL import Image, ImageOps
from cv2 import erode, dilate
from torchvision.transforms.functional import adjust_brightness, adjust_contrast

class BrightnessAdjust:
    """
    Adjust brightness
    """
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return adjust_brightness(x, self.factor)

class ContrastAdjust:
    """
    Adjust contrast
    """
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return adjust_contrast(x, self.factor)

class SignFlipping:
    """
    Color inversion
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return ImageOps.invert(x)

class DPIAdjusting:
    """
    Resolution modification
    """
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        w, h = x.size
        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)

class Dilation:
    """
    OCR: stroke width increasing
    """
    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))

class Erosion:
    """
    OCR: stroke width decreasing
    """
    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))

class ElasticDistortion:
    """
    Elastic Distortion adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    def __init__(self, grid, magnitude, min_sep):
        self.grid_width, self.grid_height = grid
        self.xmagnitude, self.ymagnitude = magnitude
        self.min_h_sep, self.min_v_sep = min_sep

    def __call__(self, x):
        w, h = x.size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [[(0, 0) for x in range(horizontal_tiles)] for y in range(vertical_tiles)]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

                sm_h = min(self.xmagnitude, width_of_square - (self.min_h_sep + shift[vertical_tile][horizontal_tile - 1][0])) if horizontal_tile > 0 else self.xmagnitude
                sm_v = min(self.ymagnitude, height_of_square - (self.min_v_sep + shift[vertical_tile - 1][horizontal_tile][1])) if vertical_tile > 0 else self.ymagnitude

                dx = random.randint(-sm_h, self.xmagnitude)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        shift = list(itertools.chain.from_iterable(shift))

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2, x3, y3, x4, y4 in dimensions:
            polygons.append([x1, y1, x2, y2, x3 + dx, y3 + dy, x4, y4])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for id, (a, b, c, d) in enumerate(polygon_indices):
            dx = shift[id][0]
            dy = shift[id][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1, x2, y2, x3 + dx, y3 + dy, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1, x2 + dx, y2 + dy, x3, y3, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1, x2, y2, x3, y3, x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy, x2, y2, x3, y3, x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        self.generated_mesh = generated_mesh

        return x.transform(x.size, Image.MESH, self.generated_mesh, resample=Image.BICUBIC)

class SaltAndPepperNoise:
    """
    Adds salt and pepper noise to the image.
    """
    def __init__(self, amount=0.05, salt_vs_pepper=0.5):
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def __call__(self, x):
        img = np.array(x)
        row, col, ch = img.shape
        s_vs_p = self.salt_vs_pepper
        amount = self.amount
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[coords[0], coords[1], :] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[coords[0], coords[1], :] = 0
        return Image.fromarray(out)

class PerspectiveTransform:
    """
    Applies a perspective transformation to the image.
    """
    def __init__(self, val):
        self.val = val

    def __call__(self, x):
        w, h = x.size

        def rd(d):
            return random.uniform(-d, d)

        # Generate a random projective transform
        tl_top = rd(self.val)
        tl_left = rd(self.val)
        bl_bottom = rd(self.val)
        bl_left = rd(self.val)
        tr_top = rd(self.val)
        tr_right = rd(self.val)
        br_bottom = rd(self.val)
        br_right = rd(self.val)

        tform = stf.ProjectiveTransform()
        tform.estimate(np.array((
            (tl_left, tl_top),
            (bl_left, h - bl_bottom),
            (w - br_right, h - br_bottom),
            (w - tr_right, tr_top)
        )), np.array((
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        )))

        # Determine shape of output image, to preserve size
        corners = np.array([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minr + 1
        output_shape = np.around((out_rows, out_cols))

        # Fit output image in new shape
        translation = (minc, minr)
        tform4 = stf.SimilarityTransform(translation=translation)
        tform = tform4 + tform
        tform.params /= tform.params[2, 2]

        x = stf.warp(np.array(x), tform, output_shape=output_shape, cval=255, preserve_range=True)
        x = stf.resize(x, (h, w), preserve_range=True).astype(np.uint8)

        return Image.fromarray(x)


# Example usage of augmentation methods
image_path = 'augumented_images/blurred.png'
original_image = Image.open(image_path)

# Apply augmentations
brightness_adjust = BrightnessAdjust(1.5)(original_image)
contrast_adjust = ContrastAdjust(1.5)(original_image)
sign_flipping = SignFlipping()(original_image)
dpi_adjusting = DPIAdjusting(1.5)(original_image)
dilation = Dilation((5, 5), 1)(original_image)
erosion = Erosion((5, 5), 1)(original_image)
# elastic_distortion = ElasticDistortion((10, 10), (5, 5), (2, 2))(original_image)
salt_and_pepper_noise = SaltAndPepperNoise(0.1, 0.5)(original_image)
perspective_transform = PerspectiveTransform(0.2)(original_image)

# Save or display the results as needed
brightness_adjust.save('/Users/elona/Documents/layout-analysis-OMR/new_transform/brightness_adjusted.png')
contrast_adjust.save('/Users/elona/Documents/layout-analysis-OMR/new_transform/contrast_adjusted.png')
sign_flipping.save('/Users/elona/Documents/layout-analysis-OMR/new_transform/sign_flipped.png')
dpi_adjusting.save('/Users/elona/Documents/layout-analysis-OMR/new_transform/dpi_adjusted.png')
dilation.save('/Users/elona/Documents/layout-analysis-OMR/new_transform/dilated.png')
erosion.save('/Users/elona/Documents/layout-analysis-OMR/new_transform/eroded.png')
# elastic_distortion.save('/Users/elona/Documents/layout-analysis-OMR/new_transform/elastic_distorted.png')
salt_and_pepper_noise.save('/Users/elona/Documents/layout-analysis-OMR/new_transform/salt_and_pepper.png')
perspective_transform.save('/Users/elona/Documents/layout-analysis-OMR/new_transform/perspective_transformed.png')
