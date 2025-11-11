import cv2
import numpy as np
import os
import extract_descriptors
import cvpr_compare
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import config


images_folder = os.path.join(config.dataset_folder, config.dataset_images_subfolder)
query_image_path = os.path.join(images_folder, "12_5_s.bmp")  # Test query image


def compute_histogram_descriptor(image_path, bins=6):
    
    image = cv2.imread(image_path).astype(np.float64)
    if image is None:
        return None
    hist =  extract_descriptors.color_histogram(image, num_bins=bins)
    return hist

def main():
    # Load query image and compute descriptor
    query_image = cv2.imread(query_image_path)
    if query_image is None:
        print(f"Error: Could not load query image {query_image_path}")
        return
    
    query_descriptor = compute_histogram_descriptor(query_image_path, bins=6)
    if query_descriptor is None:
        print("Error: Could not compute query descriptor")
        return
    
    print(f"Query image: {os.path.basename(query_image_path)}")
    print(f"Descriptor shape: {query_descriptor.shape}")
    print()
    
    # Get all image files from the Images folder
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.bmp')])
    
    # Compute distances to all images
    distances = []
    descriptors = []
    
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        descriptor = compute_histogram_descriptor(image_path, bins=6)
        
        if descriptor is not None:
            # Compute chi-squared distance
            distance = cvpr_compare.dist_chi_squared(query_descriptor, descriptor)
            distances.append(distance)
            descriptors.append(descriptor)
        else:
            distances.append(float('inf'))
            descriptors.append(None)
    
    # Sort by distance to get ranked results
    sorted_indices = np.argsort(distances)
    
    # Skip the first result (query itself) and get top 11 results
    top_11_indices = sorted_indices[1:12]
    top_11_paths = [os.path.join(images_folder, image_files[i]) for i in top_11_indices]
    
    # Print top 11 results with filenames and ranks
    print("=" * 60)
    print("Top 11 Results (Chi-squared Distance) - Excluding Query Image")
    print("=" * 60)
    print(f"{'Rank':<6}{'Filename':<20}{'Distance':<15}")
    print("-" * 60)
    
    for rank, idx in enumerate(top_11_indices, 1):
        filename = image_files[idx]
        distance = distances[idx]
        print(f"{rank:<6}{filename:<20}{distance:<15.6f}")
    
    print("=" * 60)
    print()
    
    # Display query image and top 11 results (excluding query itself)
    num_results = 11
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle(f'Global Color Histogram; Bins=6; Top {num_results} Results (Chi-squared Distance)', fontsize=12)
    
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Query image in first position
    axes[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Query\n{os.path.basename(query_image_path)}')
    axes[0].axis('off')
    
    # Top 11 results
    for i, (rank, idx) in enumerate(zip(range(1, num_results + 1), top_11_indices)):
        ax_idx = i + 1
        image_path = os.path.join(images_folder, image_files[idx])
        result_image = cv2.imread(image_path)
        if result_image is not None:
            axes[ax_idx].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            distance = distances[idx]
            axes[ax_idx].set_title(f'Rank {rank}\n{image_files[idx]}\nDist: {distance:.4f}')
        axes[ax_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('hist_search_results.png', dpi=100, bbox_inches='tight')
    print("Results saved to hist_search_results.png")
    plt.show()

if __name__ == "__main__":
    main()
