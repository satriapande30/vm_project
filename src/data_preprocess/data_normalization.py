import cv2
import os

TARGET_SIZE = (640, 480)
DATASET_PATH = "D:/KULIAH/SKRIPSI/vm_project/data/raw"
PROCESSED_DATASET_PATH = "D:/KULIAH/SKRIPSI/vm_project/data/processed_dataset/"

if not os.path.exists(PROCESSED_DATASET_PATH):
    os.makedirs(PROCESSED_DATASET_PATH)

for class_folder in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_folder)
    if os.path.isdir(class_path):
        file_count = 0
        for image_file in sorted(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, TARGET_SIZE)
            output_path = os.path.join(PROCESSED_DATASET_PATH, class_folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            new_file_name = f"{class_folder}_{file_count}.jpg"
            output_image_path = os.path.join(output_path, new_file_name)
            cv2.imwrite(output_image_path, image)
            file_count += 1