import numpy as np


def avg_rgb(img):
    # Compute the average RGB color of the image
    avg_color = np.mean(img, axis=(0, 1))
    return_value = avg_color.reshape(1, -1)  # Return as a row vector 
    return return_value


def histogram(image, num_bins=4):
    """
    Compute a global color histogram for the given image.

    Parameters:
    - image: Input image in BGR format (as read by OpenCV).
    - num_bins: Number of bins for each color channel.

    Returns:
    - hist: Flattened histogram as a 1D numpy array.
    """
    
    # Convert image to float32 for better precision
    image = image.astype(np.float32) / 256.0

    # image = image[100:104, 100:105, :]
    image_rgb_bins = np.floor(image * num_bins).astype(int)

    image_bins = (image_rgb_bins[:, :, 0] * (num_bins ** 2) +
            image_rgb_bins[:, :, 1] * num_bins +
            image_rgb_bins[:, :, 2])

    image_histogram = np.histogram(image_bins, bins=num_bins**3)

    hist = image_histogram[0]
    hist = hist/np.sum(hist)  # Normalize to sum to 1
    return hist





def histogram_normalized(image, num_bins=4):
    """
    Compute a global color histogram for the given image.

    Parameters:
    - image: Input image in BGR format (as read by OpenCV).
    - num_bins: Number of bins for each color channel.

    Returns:
    - hist: Flattened histogram as a 1D numpy array.
    """
    # Convert image to float32 for better precision
    image = image.astype(np.float32) / 256.0

    image_bgr_bins = np.floor(image * num_bins)

    image_bins = (image_bgr_bins[:, :, 0] * (num_bins ** 2) +
            image_bgr_bins[:, :, 1] * num_bins +
            image_bgr_bins[:, :, 2])

    image_histogram = np.histogram(image_bins, bins=num_bins**3)

    hist = image_histogram[0]
    hist = hist/np.sum(hist)  # Normalize to sum to 1
    return hist
