# verify_golden_set.py

import os
import cv2

def verify_golden_set(images_dir, labels_dir):
    """
    Verify the integrity and format of the golden test set images and labels.

    Args:
        images_dir (str): Path to the directory containing golden test images.
        labels_dir (str): Path to the directory containing golden test labels.

    Returns:
        bool: True if all images and labels are valid, False otherwise.
    """
    # Check if directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("Images or labels directory does not exist.")
        return False

    # Verify each image and its corresponding label
    for image_file in os.listdir(images_dir):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            label_file = os.path.splitext(image_file)[0] + '.txt'  # Assuming labels are in .txt format
            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, label_file)

            # Check if the image can be read
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image {image_file} is not valid.")
                return False

            # Check if the corresponding label file exists
            if not os.path.isfile(label_path):
                print(f"Label file for {image_file} does not exist.")
                return False

    print("All images and labels are valid.")
    return True

if __name__ == "__main__":
    images_directory = 'data/test/golden_test_set/images'
    labels_directory = 'data/test/golden_test_set/labels'
    verify_golden_set(images_directory, labels_directory)