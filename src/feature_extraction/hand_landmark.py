import cv2
import mediapipe as mp
import os
import csv
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime

class HandLandmarkExtractor:
    def __init__(self, dataset_path, output_file, batch_size=1000):
        self.dataset_path = dataset_path
        self.output_file = output_file
        self.batch_size = batch_size
        self.mp_hands = mp.solutions.hands
        self.logger = self._setup_logger()
        self.detected_hands = 0
        self.undetected_hands = 0

    def _setup_logger(self):
        log_dir = 'output/log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"hand_landmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        return logger

    def extract_features(self):
        self.logger.info("Starting feature extraction process")
        
        with open(self.output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["label"]
            for i in range(21):
                header.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])
            writer.writerow(header)

            class_folders = [f for f in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, f))]
            for class_folder in tqdm(class_folders, desc="Processing classes"):
                class_path = os.path.join(self.dataset_path, class_folder)
                self.logger.info(f"Processing class: {class_folder}")
                self._process_class(class_path, class_folder, writer)

        total_images = self.detected_hands + self.undetected_hands
        detection_rate = (self.detected_hands / total_images) * 100 if total_images > 0 else 0
        
        self.logger.info(f"Total images processed: {total_images}")
        self.logger.info(f"Hands detected: {self.detected_hands}")
        self.logger.info(f"Hands not detected: {self.undetected_hands}")
        self.logger.info(f"Hand detection rate: {detection_rate:.2f}%")
        
        print(f"Hand detection rate: {detection_rate:.2f}%")
        
        self.logger.info("Feature extraction process completed")

    def _process_class(self, class_path, class_label, writer):
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(image_files)
        
        with tqdm(total=total_images, desc=f"Extracting {class_label}", leave=False) as pbar:
            for i in range(0, total_images, self.batch_size):
                batch = image_files[i:i+self.batch_size]
                self.logger.info(f"Processing batch {i//self.batch_size + 1} of class {class_label}")
                
                with self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                    for image_file in batch:
                        image_path = os.path.join(class_path, image_file)
                        self._process_image(image_path, class_label, hands, writer)
                        pbar.update(1)
                
                self.logger.info(f"Completed {min(i+self.batch_size, total_images)}/{total_images} images for class {class_label}")

    def _process_image(self, image_path, class_label, hands, writer):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Unable to read image: {image_path}")
            
            image = cv2.resize(image, (640, 480))
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_values = [class_label]
                for landmark in hand_landmarks.landmark:
                    landmark_values.extend([landmark.x, landmark.y, landmark.z])
                writer.writerow(landmark_values)
                self.logger.debug(f"Extracted landmarks for {image_path}")
                self.detected_hands += 1
            else:
                self.logger.warning(f"No hand detected in {image_path}")
                self.undetected_hands += 1
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            self.undetected_hands += 1

if __name__ == "__main__":
    DATASET_PATH = "D:/KULIAH/SKRIPSI/vm_project/data/"
    OUTPUT_FILE = "D:/KULIAH/SKRIPSI/vm_project/output/features/hand_landmark_features.csv"
    
    extractor = HandLandmarkExtractor(DATASET_PATH, OUTPUT_FILE)
    extractor.extract_features()