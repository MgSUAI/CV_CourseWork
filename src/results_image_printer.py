import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import config


def load_and_resize(path, size):
    """Load an image and resize it to the target size."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return cv2.resize(img, size)


def add_label_inside(img, label):
    """add label inside the image."""
    font_scale = 1.5
    font_thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX

    labeled = img.copy()
    h, w, _ = labeled.shape
    text = str(label)
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Position text near bottom center
    x = (w - text_width) // 2
    y = h - 15  # 15 pixels from bottom
    
    # Draw a semi-transparent rectangle behind text for visibility
    overlay = labeled.copy()
    cv2.rectangle(overlay, (x - 10, y - text_height - 5), (x + text_width + 10, y + 10), (255, 255, 255), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, labeled, 1 - alpha, 0, labeled)
    
    # Draw black text on top
    cv2.putText(labeled, text, (x, y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    return labeled

def image_printer(image_groups):
    """prints query image and it's top search results for each class"""
    image_size = (200, 150)  
    gap_size = 20             # horizontal gap after first/query image
    row_gap = 5              # vertical gap between classes

    # build each row 
    row_images = []
    for group in image_groups:
        labeled_images = []
        for path, number in group:
            try:
                img = load_and_resize(path, image_size)
                img_labeled = add_label_inside(img, number)
                labeled_images.append(img_labeled)
            except FileNotFoundError as e:
                print(e)

        if not labeled_images:
            continue

        # Add horizontal gap after first image
        gap = np.ones((image_size[1], gap_size, 3), dtype=np.uint8) * 255
        if len(labeled_images) > 1:
            images_with_gap = [labeled_images[0], gap] + labeled_images[1:]
        else:
            images_with_gap = labeled_images

        row_img = np.hstack(images_with_gap)
        row_images.append(row_img)

    # stack rows vertically
    if len(row_images) > 1:
        vgap = np.ones((row_gap, row_images[0].shape[1], 3), dtype=np.uint8) * 255
        stacked = []
        for i, row in enumerate(row_images):
            stacked.append(row)
            if i < len(row_images) - 1:
                stacked.append(vgap)
        final_img = np.vstack(stacked)
    else:
        final_img = row_images[0]

    # print images
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(16, 10))
    plt.figure(figsize=(16, 13))
    plt.imshow(final_rgb)
    plt.axis("off")
    plt.title("Images (300x200) with Labels Inside")
    plt.savefig("plt_Results.png", dpi=300, bbox_inches="tight")
    
    if config.display_plots:
        plt.show()

    print("Search result sample saved in plt_Results.png")


if __name__ == '__main__':

    image_folder = os.path.join(config.dataset_folder, config.dataset_images_subfolder)

    image_files_temp = sorted(glob.glob(f"{image_folder}/*.bmp"))
    image_files = [(img_path, np.int16(os.path.basename(img_path).split('_')[0])) for img_path in image_files_temp]

    image_groups = []
    num_images_per_group = 10
    for i in range(20):
        img_list = image_files[i*num_images_per_group : (i+1)*num_images_per_group]
        image_groups.append(img_list)

    print(len(image_groups[0]))
    image_printer(image_groups)