import os
import numpy as np
import cv2
import scipy.io as sio
# import matplotlib
import extract_descriptors
from tqdm import tqdm
import config


def compute_descriptors(descriptor_type, **kwargs):
    """
    Compute and save image descriptors for all images in the dataset.
    """

    print(f"Computing {descriptor_type} descriptors...")
    image_folder = os.path.join(config.dataset_folder, config.dataset_images_subfolder)
    output_folder = os.path.join(config.output_folder, config.output_subfolder)

    for filename in tqdm(os.listdir(image_folder), desc="    Processing"):
        if filename.endswith(".bmp"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path).astype(np.float64) # todo: without normalize the image
            output_file = os.path.join(output_folder, filename.replace('.bmp', '.mat'))
            
            if descriptor_type == 'avg_rgb':
                F = extract_descriptors.avg_rgb(image)
            elif descriptor_type == 'color_histogram':
                F = extract_descriptors.color_histogram(image, kwargs.get('num_bins'))

            elif descriptor_type == 'compute_grid_color_histogram':
                F = extract_descriptors.compute_grid_color_histogram(image
                                                                     , kwargs.get('num_bins')
                                                                     , kwargs.get('grid_rows') 
                                                                     , kwargs.get('grid_cols'))
            else:
                raise ValueError(f"Unknown descriptor type: {descriptor_type}")
            
            # Save the descriptor to a .mat file
            sio.savemat(output_file, {'F': F})
