import numpy as np
import cv2


def avg_rgb(image_path):
    """
    Computes the average RGB color of the given image.

    Args:
        img (numpy.ndarray): A 3D array representing the image, 
                            where the dimensions correspond to height, width, and color channels.

    Returns:
        numpy.ndarray: A 2D array (1 x 3) containing the average RGB values of the image.
    """
    image = cv2.imread(image_path).astype(np.float64) 
    # Compute the average RGB color of the image
    avg_color = np.mean(image, axis=(0, 1))
    return_value = avg_color.reshape(1, -1)  # Return as a row vector 
    return return_value


def color_histogram(image, num_bins=4):
    """
    Computes a global color histogram for the given image.
    The function computes a normalized color histogram by quantizing the color space
    into the specified number of bins for each channel.
    Args:
        image: np.ndarray
            Input image in BGR format (as read by OpenCV)
        num_bins: int, optional
            Number of bins for each color channel (default is 4)
            The total number of bins will be num_bins^3
        np.ndarray
            Normalized histogram as a 1D array of length num_bins^3
    """
    # image = cv2.imread(image).astype(np.float64) 

    image_rgb_bins = np.floor(image * num_bins).astype(int)

    image_bins = (image_rgb_bins[:, :, 0] * (num_bins ** 2) +
            image_rgb_bins[:, :, 1] * num_bins +
            image_rgb_bins[:, :, 2])

    image_histogram = np.histogram(image_bins, bins=num_bins**3)

    hist = image_histogram[0]
    hist = hist/np.sum(hist)  # Normalize to sum to 1
    return hist



def compute_grid_color_histogram(image_path, bins=4, grid_rows=4, grid_cols=4):
    """
    Compute a concatenated grid-wise color histogram for an image.

    Args:
        image (numpy.ndarray): Input color image in BGR format.
        bins (int): Number of bins per color channel for each cell histogram. 
        grid_rows (int): Number of rows in the spatial grid.
        grid_cols (int): Number of columns in the spatial grid.

    Returns:
        numpy.ndarray: 1-D feature vector containing the concatenated per-cell color histograms.
    """

    image = cv2.imread(image_path).astype(np.float64) 
    h, w, _ = image.shape

    # Grid cell size
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    feature_vector = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            cell = image[y0:y1, x0:x1]

            # hist = cv2.calcHist([cell], [0, 1, 2], None,
            #                     [bins, bins, bins],
            #                     [0, 256, 0, 256, 0, 256])
            
            hist = color_histogram(cell, num_bins=bins)
            hist = hist.flatten()
            hist = hist / np.sum(hist)  # Normalize
            feature_vector.extend(hist)

    return np.array(feature_vector)


def compute_eoh(image_path, eoh_bins, grid_rows, grid_cols):

    # Read and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception(f"Could not read {image_path}")

    # Compute gradients using Sobel operators
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=True)  # angle in [0, 360)
    angle = np.mod(angle, 180)  # reduce to [0, 180) since direction is symmetrical

    h, w = image.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    feature_vector = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w

            cell_mag = magnitude[y0:y1, x0:x1]
            cell_ang = angle[y0:y1, x0:x1]

            hist, _ = np.histogram(cell_ang,
                                   bins=eoh_bins,
                                   range=(0, 180),
                                   weights=cell_mag)
            hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
            feature_vector.extend(hist)

    return np.array(feature_vector)

