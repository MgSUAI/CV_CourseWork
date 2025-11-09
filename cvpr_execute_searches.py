import config
import os
from random import randint
import numpy as np
import scipy.io as sio
import scipy.stats as stats
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from types import SimpleNamespace
import cvpr_computedescriptors # import compute_descriptors, compute_pca
import cvpr_compare
import results_image_printer


def get_class_from_filename(filename):
    """
    Extract class label from the filename.

    Args:
        filename: filename from which to extract the class label
        
    Returns:
        np.int16: The class label as a 16-bit integer.
    """
    base = os.path.basename(filename)
    return np.int16(base.split('_')[0])


def get_index_per_class(labels):

    unique_labels = np.unique(labels)

    # For each unique number, pick one random index where it occurs
    selected_indices = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]     # all indices for this value
        chosen = np.random.choice(indices)         # pick one randomly
        selected_indices.append(chosen)

    # Convert to NumPy array
    selected_indices = np.array(selected_indices)
    return selected_indices


def get_image_result_row(result_indices, files, labels):
    image_group_row = []

    for index in result_indices.tolist():
        image_group_row.append((files[index], labels[index]))

    return image_group_row


def load_dataset(run_configs):
    """
    Load image descriptors, file paths, and class labels from the dataset.

    Returns:
        tuple: A tuple containing three numpy arrays:
            - all_files (np.ndarray): Array of image file paths
            - all_labels (np.ndarray): Array of class labels
            - all_features (list): List of image descriptors
    """
    all_features, all_files, all_labels = [], [], []

    # for filename in os.listdir(os.path.join(config.output_folder, config.output_subfolder)):
    #     if filename.endswith('.mat'):
    #         image_descriptor_path = os.path.join(config.output_folder, config.output_subfolder, filename)
    #         image_descriptor_data = sio.loadmat(image_descriptor_path)
    #         image_path = os.path.join(config.dataset_folder, config.dataset_images_subfolder, filename.replace('.mat', '.bmp'))  

    #         all_files.append(image_path)
    #         all_features.append(image_descriptor_data['F'][0])  # F is a 1D array
    #         all_labels.append(get_class_from_filename(filename))

    descriptor_list = cvpr_computedescriptors.compute_descriptors(run_configs)
    for descriptor in descriptor_list:
        file_path, feature = descriptor
        all_files.append(file_path)
        all_features.append(feature)
        all_labels.append(get_class_from_filename(os.path.basename(file_path)))

    return np.array(all_files), np.array(all_labels), all_features


def run_image_search(query_index, labels, features, print_query_indices, run_configs, cov_inv = None):
    """
    Perform image search using global histogram features and compute precision-recall metrics.

    Args:
        query_index (int): Index of the query image
        labels (np.ndarray): Array of class labels for all images
        features (list): List of feature descriptors for all images

    Returns:
        tuple: A tuple containing:
            - precisions (np.ndarray): Array of precision values
            - recalls (np.ndarray): Array of recall values
            - predicted_label (int): Predicted class label based on top-5 results
            - print_query_indices (np.ndarray): Array of one index per image class, used to print results
    """
    # Compute the distance between the query and all other descriptors

    distance_list = []
    query_feature = features[query_index]
    query_label = labels[query_index]

    for i in range(len(features)):

        # if config.distance_metric == 'manhattan':
        #     distance = cvpr_compare.dist_manhattan(query_feature, features[i])        
        # elif config.distance_metric == 'euclidean':
        #     distance = cvpr_compare.dist_euclidean(query_feature, features[i])
        # elif config.distance_metric == 'mahalanobis':
        #     distance = cvpr_compare.dist_mahalanobis(query_feature, features[i], cov_inv)
        # elif config.distance_metric == 'chi_squared':
        #     distance = cvpr_compare.dist_chi_squared(query_feature, features[i])
        if run_configs.distance_metric == 'manhattan':
            distance = cvpr_compare.dist_manhattan(query_feature, features[i])        
        elif run_configs.distance_metric == 'euclidean':
            distance = cvpr_compare.dist_euclidean(query_feature, features[i])
        elif run_configs.distance_metric == 'mahalanobis':
            distance = cvpr_compare.dist_mahalanobis(query_feature, features[i], cov_inv)
        elif run_configs.distance_metric == 'chi_squared':
            distance = cvpr_compare.dist_chi_squared(query_feature, features[i])


        # is_same_class = os.path.basename(all_files[query_image_index]).split('_')[0] == os.path.basename(all_files[i]).split('_')[0]
        distance_list.append(distance)

    sorted_indices = np.argsort(distance_list)[1:]  # Exclude the query itself
    predicted_labels = labels[sorted_indices]

    if query_index in print_query_indices:
        # print_row = np.array(query_index)
        print_row = np.append([query_index], sorted_indices[:config.num_result_images_to_print])
    else:
        print_row = None

    relevances = np.array([1 if label == query_label else 0 for label in predicted_labels])

    total_relevant = np.sum(labels == query_label) - 1  # Exclude the query itself
    if total_relevant <= 1:
        return None  # skip classes with only one image

    precisions, recalls = [], []
    retrieved_relevant = 0

    for i, rel in enumerate(relevances, start=1):
        if rel == 1:
            retrieved_relevant += 1
        precision = retrieved_relevant / i
        recall = retrieved_relevant / (total_relevant) 
        precisions.append(precision)
        recalls.append(recall)

    # use the mode of first 10 predicted labels as the final predicted label
    predicted_label = stats.mode(predicted_labels[0:10])[0]
    return np.array(precisions), np.array(recalls), predicted_label, print_row


def show_images_matplot(image_paths):
    """
    Display multiple images in a horizontal grid using matplotlib.

    Args:
        image_paths (list): List of file paths to the images to be displayed
    """
    # Create a figure
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))

    # Loop through the images and plot each one
    for i, ax in enumerate(axes):
        img = mpimg.imread(image_paths[i])
        ax.imshow(img)
        ax.axis('off')  # Hide axes for better visualization

        filename = os.path.splitext(os.path.basename(image_paths[i]))[0]
        ax.set_title(filename, fontsize=10, pad=10)

    # Adjust layout for better spacing
    plt.tight_layout()
    if config.display_plots:
        plt.show()


def remove_descriptor_files():
    """
    Remove all .mat descriptor files from the output directory.
    """
    print("Removing existing descriptor files...")
    descriptor_folder = os.path.join(config.output_folder, config.output_subfolder)
    for filename in os.listdir(descriptor_folder):
        if filename.endswith('.mat'):
            file_path = os.path.join(descriptor_folder, filename)
            os.remove(file_path)


def plot_pr_curves(metric_results, output_folder):
    """
    Plot and save the precision-recall curves for multiple distance metrics on the same plot.

    Args:
        metric_results (dict): Dictionary containing results for each metric with structure:
            {
                'metric_name': {
                    'recall_points': np.ndarray,
                    'mean_precision': np.ndarray,
                    'mean_average_precision': float
                }
            }
        output_folder (str): Directory where the plot image will be saved

    Returns:
        None: Saves the plot as 'plt_pr_curves_comparison.png' in the given output folder
    """
    colors = {
        'manhattan': 'blue',
        'euclidean': 'red',
        'chi_squared': 'green',
        'mahalanobis': 'purple'
    }
    
    plt.figure(figsize=(10, 8))
    
    for metric_name, results in metric_results.items():
        color = colors.get(metric_name, None)
        plt.plot(results['recall_points'], 
                 results['mean_precision'], 
                 color=color, 
                 label=f"{metric_name} (mAP = {results['mean_average_precision']:.3f})")

    plt.axis([0, 1, 0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Mean Precision–Recall Curves Comparison\n({run_configs.descriptor_type if 'run_configs' in globals() else ''})")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(output_folder, "plt_pr_curves_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if config.display_plots:
        plt.show()

    print(f"Combined Precision–Recall curves saved as {out_path}")


def plot_confusion_matrix(y_true, y_pred, unique_classes):
    """
    Create and save a confusion matrix visualization for classification results.

    Args:
        y_true (list): List of true class labels
        y_pred (list): List of predicted class labels
        unique_classes (list): List of all unique class labels

    Returns:
        None: Saves the plot as 'plt_confusion_matrix.png' and displays it
    """
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)

    disp.plot(cmap='Reds', xticks_rotation=45, colorbar=True)
    plt.title("Confusion Matrix (Top-5 Predicted Class per Query)")
    plt.savefig("plt_confusion_matrix.png", dpi=300, bbox_inches="tight")
    
    if config.display_plots:
        plt.show()

    print("Confusion matrix saved as plt_confusion_matrix.png")


def main(run_configs):
    """
    Main function that orchestrates the image retrieval and evaluation process.
    
    This function:
    1. Sets up the output directory
    2. Computes image descriptors
    3. Performs image search using multiple distance metrics
    4. Computes and plots combined precision-recall metrics
    5. Creates and saves confusion matrix for best performing metric
    
    Returns:
        None
    """
    output_descriptor_folder = os.path.join(config.output_folder, config.output_subfolder)
    output_plots_folder = os.path.join(config.output_folder, config.output_plots_subfolder)
    output_metrics_folder = os.path.join(config.output_folder, config.output_metrics_subfolder)
    os.makedirs(output_descriptor_folder, exist_ok=True)
    os.makedirs(output_plots_folder, exist_ok=True)
    os.makedirs(output_metrics_folder, exist_ok=True)
    remove_descriptor_files()

    # Load dataset once
    all_files, all_labels, all_features = load_dataset(run_configs)
    
    # Apply PCA if configured
    if config.use_pca:
        all_features = cvpr_computedescriptors.compute_pca(all_features, run_configs)
    
    # Always compute covariance matrix for Mahalanobis distance
    cov_inv = cvpr_computedescriptors.compute_covariance_matrix(all_features)

    print_query_indices = get_index_per_class(all_labels)
    
    # Dictionary to store results for each distance metric
    metric_results = {}
    distance_metrics = ['manhattan', 'euclidean', 'chi_squared', 'mahalanobis']
    best_map = -1
    best_metric = None
    best_results = None  # To store results for confusion matrix

    # Run evaluation for each distance metric
    for metric in distance_metrics:
        print(f"\nEvaluating {metric} distance metric...")
        run_configs.distance_metric = metric
        
        all_precisions, all_recalls, y_true, y_pred = [], [], [], []
        query_result_print_data = []
        
        for index in tqdm(range(len(all_files)), desc="     Calculating: "):
            precisions, recalls, predicted_label, print_row = run_image_search(index, all_labels, all_features, print_query_indices, run_configs, cov_inv)
            
            if precisions is not None:  # Skip if no results
                all_precisions.append(precisions)
                all_recalls.append(recalls)
                y_true.append(all_labels[index])
                y_pred.append(predicted_label)

                if print_row is not None and metric == distance_metrics[0]:  # Only print for first metric
                    query_result_print_data.append(get_image_result_row(print_row, all_files, all_labels))

        # Calculate average precision-recall
        recall_points = np.linspace(0, 1.0, 11)
        interp_precisions = []

        for precisions, recalls in zip(all_precisions, all_recalls):
            interp = np.interp(recall_points, recalls, precisions)
            interp_precisions.append(interp)

        mean_precision = np.mean(interp_precisions, axis=0)
        mean_average_precision = np.mean([np.mean(p) for p in interp_precisions])

        metric_results[metric] = {
            'recall_points': recall_points,
            'mean_precision': mean_precision,
            'mean_average_precision': mean_average_precision
        }

        # Keep track of best performing metric for confusion matrix
        if mean_average_precision > best_map:
            best_map = mean_average_precision
            best_metric = metric
            best_results = (y_true, y_pred)

        # Print results for each metric
        print(f"{metric} Mean Average Precision (mAP): {mean_average_precision:.4f}")

    # Print results from first metric run
    if query_result_print_data:
        results_image_printer.image_printer(query_result_print_data)

    # Plot combined PR curves
    plot_pr_curves(metric_results, output_plots_folder)
    
    # Plot confusion matrix for best performing metric
    print(f"\nGenerating confusion matrix for best performing metric: {best_metric}")
    unique_classes = sorted(list(set(all_labels)))
    plot_confusion_matrix(best_results[0], best_results[1], unique_classes)


if __name__ == '__main__':

    run_configs = {
        'descriptor_type': 'grid_color_histogram',
        'num_bins': 4,
        'grid_rows': 8,
        'grid_cols': 8,
        'distance_metric': 'euclidean'
    }

    run_configs = SimpleNamespace(**run_configs)
    main(run_configs)
1   