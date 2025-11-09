import os
import numpy as np
import cv2
import scipy.io as sio
import extract_descriptors
from glob import glob
from tqdm import tqdm
from sklearn.decomposition import PCA
import config


def compute_descriptors(run_configs):
    """
    Compute and save image descriptors for all images in the dataset.
    """

    print(f"Computing {run_configs.descriptor_type} descriptors...")
    image_folder = os.path.join(config.dataset_folder, config.dataset_images_subfolder)
    output_folder = os.path.join(config.output_folder, config.output_subfolder)
    descriptor_list = []

    for image_path in tqdm(sorted(glob(f"{image_folder}/*.bmp")), desc="    Processing"):
        if image_path.endswith(".bmp"):
            # image_path = os.path.join(image_folder, os.path.basename(filename))



            image = cv2.imread(image_path).astype(np.float64) 
            output_file = os.path.join(output_folder, os.path.basename(image_path).replace('.bmp', '.mat'))
            
            if  run_configs.descriptor_type == 'avg_rgb':
                F = extract_descriptors.avg_rgb(image_path)
            elif run_configs.descriptor_type == 'global_color_histogram':
                F = extract_descriptors.color_histogram(image, run_configs.num_bins)

            elif run_configs.descriptor_type == 'grid_color_histogram':
                F = extract_descriptors.compute_grid_color_histogram(image_path,
                                                                     run_configs.num_bins,
                                                                     run_configs.grid_rows, 
                                                                     run_configs.grid_cols)
            elif run_configs.descriptor_type == 'eoh':
                F = extract_descriptors.compute_eoh(image_path,
                                                    run_configs.eoh_bins,
                                                    run_configs.grid_rows, 
                                                    run_configs.grid_cols)
            elif run_configs.descriptor_type == 'eoh_plus_ch':
                F_gch = extract_descriptors.compute_grid_color_histogram(image_path,
                                                                     run_configs.gch_bins,
                                                                     run_configs.grid_rows, 
                                                                     run_configs.grid_cols)
                F_eoh = extract_descriptors.compute_eoh(image_path,
                                                    run_configs.eoh_bins,
                                                    run_configs.grid_rows, 
                                                    run_configs.grid_cols)

                F_gch = F_gch * run_configs.weight_color
                F_eoh = F_eoh * run_configs.weight_eoh
                F = np.concatenate([F_gch, F_eoh])

            else:
                raise ValueError(f"Unknown descriptor type: {run_configs.descriptor_type}")
            
            # Save the descriptor to a .mat file
            sio.savemat(output_file, {'F': F})
            descriptor_list.append((image_path, F))

    return descriptor_list            


def compute_pca(descriptors):
    print("Computing PCA and covariance matrix")
    # Reduce descriptors to lower dimension
    pca = PCA(n_components=run_configs.pca_components)
    descriptors_pca = pca.fit_transform(descriptors)
    # query_desc_pca = pca.transform([query_desc])[0]

    # Compute covariance matrix of PCA-reduced descriptors
    cov = np.cov(descriptors_pca, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    return descriptors_pca, cov_inv


if __name__ == "__main__":
    descriptors = compute_descriptors()
    file_path, feature = descriptors[0]
    print(f"File Path = {file_path}")
    print(f"Feature Sum = {feature.sum()}")
