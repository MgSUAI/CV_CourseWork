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
from cvpr_computedescriptors import compute_descriptors
from cvpr_compare import cvpr_compare


def get_class_from_filename(filename):
    """
    Extract class label from the given filename.
    Assumes the class label is the substring before the first underscore.
    """
    base = os.path.basename(filename)
    return np.int16(base.split('_')[0])


def load_dataset():

    all_features, all_files, all_labels = [], [], []

    for filename in os.listdir(os.path.join(config.output_folder, config.output_subfolder)):
        if filename.endswith('.mat'):
            image_descriptor_path = os.path.join(config.output_folder, config.output_subfolder, filename)
            image_descriptor_data = sio.loadmat(image_descriptor_path)
            image_path = os.path.join(config.dataset_folder, config.dataset_images_subfolder, filename.replace('.mat', '.bmp'))  

            all_files.append(image_path)
            all_features.append(image_descriptor_data['F'][0])  # F is a 1D array
            all_labels.append(get_class_from_filename(filename))

    return np.array(all_files), np.array(all_labels), all_features


def image_search_global_histogram(query_index, labels, features):

    # # for each element in all_features, sum the values and store in a list 
    # sum_features = [np.sum(f) for f in all_features]
    

    # Pick a random image as the query
    # query_image_index = randint(0, all_features.shape[0] - 1)

    # todo: for test only
    # query_image_index = all_files.index('/Users/milan/Documents/SurreyUni/Sem1/ComputerVision/Labs/Lab3_5/db/MSRC_ObjCategImageDatabase_v2/Images/1_1_s.bmp') 


    # Compute the distance between the query and all other descriptors
    distance_list = []
    query_feature = features[query_index]
    query_label = labels[query_index]

    for i in range(len(features)):
        
        distance = cvpr_compare(query_feature, features[i])
        # is_same_class = os.path.basename(all_files[query_image_index]).split('_')[0] == os.path.basename(all_files[i]).split('_')[0]
        distance_list.append(distance)


    sorted_indices = np.argsort(distance_list)[1:]  # Exclude the query itself
    predicted_labels = labels[sorted_indices]

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

    # use the mode of first 5 predicted labels as the final predicted label
    predicted_label = stats.mode(predicted_labels[0:5])[0]
    return np.array(precisions), np.array(recalls), predicted_label


################






    # Show the top 5 results
    # query_image = cv2.imread(all_files[query_image_index])

    # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
    # cv2.imshow(f"Query = {os.path.basename(all_files[query_image_index])}", query_image)
    # cv2.waitKey(0)

    # num_results = 10
    # for i in range(num_results):
    #     result_image = cv2.imread(all_files[distance_list[i][1]])
    #     result_image = cv2.resize(result_image, (result_image.shape[1] // 2, result_image.shape[0] // 2))  # Make image quarter size
    #     cv2.imshow(f"Result {os.path.basename(all_files[distance_list[i][1]])}", result_image)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # result_image_paths = [all_files[query_image_index]]  # include query image first
    # for i in range(num_results):
    #     result_image_path = all_files[distance_list[i][1]]   
    #     result_image_paths.append(result_image_path) 
    # show_images_matplot(result_image_paths)
    # # show_images_cv2(result_image_paths)

    ########################

    ### xx
    # Prepare data for precision-recall curve
    # y_true = [1 if distance_list[i][2] else 0 for i in range(len(distance_list))]
    # y_scores = [distance_list[i][0] for i in range(len(distance_list))]
    # # [ distance_list[i][0] for i in range(len(distance_list))]  
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # # Calculate Average Precision (AP) score
    # ap_score = average_precision_score(y_true, y_scores)
    
    # # Plot Precision-Recall curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(recall, precision, color='b', label=f'AP = {ap_score:.2f}')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall curve')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.show()



def show_images_matplot(image_paths):
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
    plt.show()


def remove_descriptor_files():
    print("Removing existing descriptor files...")
    descriptor_folder = os.path.join(config.output_folder, config.output_subfolder)
    for filename in os.listdir(descriptor_folder):
        if filename.endswith('.mat'):
            file_path = os.path.join(descriptor_folder, filename)
            os.remove(file_path)


def plot_pr_curve(recall_points, mean_precision, mean_average_precision):
    plt.figure(figsize=(8, 6))
    plt.plot(recall_points, mean_precision, color='blue', label=f"Mean PR (mAP = {mean_average_precision:.3f})")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Mean Precision–Recall Curve (All Queries)")
    plt.legend()
    plt.grid(True)
    plt.savefig("pr_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Mean Average Precision (mAP): {mean_average_precision:.4f}")
    print("Precision–Recall curve saved as pr_curve.png")


def plot_confusion_matrix(y_true, y_pred, unique_classes):
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)

    # plt.figure(figsize=(15, 15))
    # disp.plot(cmap='Blues', xticks_rotation=45, colorbar=True)
    disp.plot(cmap='Reds', xticks_rotation=45, colorbar=True)
    plt.title("Confusion Matrix (Top-5 Predicted Class per Query)")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Confusion matrix saved as confusion_matrix.png")


def main():
    
    output_subfolder = os.path.join(config.output_folder, config.output_subfolder)
    os.makedirs(output_subfolder, exist_ok=True)
    remove_descriptor_files()

    compute_descriptors('histogram', num_bins=config.num_bins)
    all_files, all_labels, all_features = load_dataset()
    
    all_precisions, all_recalls, y_true, y_pred = [], [], [], []
    for index in range(len(all_files)):
        precisions, recalls, predicted_label = image_search_global_histogram(index, all_labels, all_features)

        all_precisions.append(precisions)
        all_recalls.append(recalls)
        y_true.append(all_labels[index])
        y_pred.append(predicted_label)

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(all_files)} queries...")

    # --- AVERAGE PRECISION-RECALL ---
    recall_points = np.linspace(0, 1.0, 11)
    interp_precisions = []

    for precisions, recalls in zip(all_precisions, all_recalls):
        interp = np.interp(recall_points, recalls, precisions)
        interp_precisions.append(interp)

    mean_precision = np.mean(interp_precisions, axis=0)
    mean_average_precision = np.mean([np.mean(p) for p in interp_precisions])
    plot_pr_curve(recall_points, mean_precision, mean_average_precision)

    unique_classes = sorted(list(set(all_labels)))
    plot_confusion_matrix(y_true, y_pred, unique_classes)



if __name__ == '__main__':
    main()

