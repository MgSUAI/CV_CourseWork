import numpy as np
import os
import cv2


def compute_histogram(image, num_bins=4):
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


def calc_hist2(image, num_bins=4):  
    """
    Calculate global color histogram of an image using OpenCV.
    
    Parameters:
        image: Input image in BGR format
        num_bins: Number of bins per channel (default=4)
    
    Returns:
        Normalized histogram as 1D numpy array
    """
    channels = [0, 1, 2]  # BGR channels
    ranges = [0, 256] * 3  # Range for each channel
    hist = cv2.calcHist([image], channels, None, [num_bins] * 3, ranges)
    hist = hist.flatten()
    hist = hist / np.sum(hist)  # Normalize
    return hist




if __name__ == "__main__":

    DATASET_FOLDER = 'db/MSRC_ObjCategImageDatabase_v2/Images'

    img_file = '2_17_s.bmp'
    img = cv2.imread(os.path.join(DATASET_FOLDER, img_file)).astype(np.float64) 
    hist = compute_histogram(img, num_bins=8)
    print(f"Histogram for {img_file}: {hist}")
    print(hist.shape)
    print(f"Sum of histogram bins: {np.sum(hist)}")  # Should be 1.0


    # hist = calc_hist2(img, num_bins=8)
    # print(f"Histogram using OpenCV for {img_file}: {hist}")
    # print(f"Sum of histogram bins: {np.sum(hist)}")  # Should be 1.0

