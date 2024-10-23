import cv2
import mediapipe as mp
import os
import csv
import logging
import numpy as np
from datetime import datetime
from tqdm import tqdm
import shutil
from imgaug import augmenters as iaa

class HandLandmarkProcessor:
    def __init__(self, dataset_path, output_file, error_dir='output/error_image', batch_size=1000):
        self.dataset_path = dataset_path
        self.output_file = output_file
        self.error_dir = error_dir
        self.batch_size = batch_size
        self.mp_hands = mp.solutions.hands
        self.logger = self._setup_logger()
        
        # Statistik dataset
        self.total_images = 0
        self.detected_hands = 0
        self.undetected_hands = 0
        self.augmented_detections = 0
        self.total_error_images = 0
        
        # Setup augmentasi
        self.augmenters = [
            iaa.Affine(rotate=(-20, 20)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.Multiply((0.8, 1.2)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
        ]
        
        # Buat direktori output
        self._setup_directories()

    def _setup_directories(self):
        """Membuat direktori yang diperlukan"""
        directories = ['output/log', self.error_dir, 'output/features']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def _setup_logger(self):
        """Setup sistem logging"""
        log_dir = 'output/log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"hand_landmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def analyze_dataset(self):
        """Menganalisis dan menghitung statistik dataset"""
        self.logger.info("Menganalisis dataset...")
        class_distribution = {}
        
        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_distribution[class_name] = num_images
                self.total_images += num_images
        
        self.logger.info(f"Total gambar dalam dataset: {self.total_images}")
        self.logger.info("Distribusi kelas:")
        for class_name, count in class_distribution.items():
            self.logger.info(f"- {class_name}: {count} gambar ({count/self.total_images*100:.2f}%)")
        
        return class_distribution

    def _try_detect_with_augmentation(self, image, hands):
        """Mencoba deteksi dengan augmentasi jika deteksi normal gagal"""
        for augmenter in self.augmenters:
            aug_image = augmenter(image=image)
            results = hands.process(cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                self.augmented_detections += 1
                return results
        return None

    def _extract_landmarks(self, image_path, class_label, hands, writer):
        """Mengekstrak landmark dari gambar dengan multiple attempt"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
            
            image = cv2.resize(image, (640, 480))
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Jika deteksi normal gagal, coba dengan augmentasi
            if not results.multi_hand_landmarks:
                results = self._try_detect_with_augmentation(image, hands)
            
            if results and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_values = [class_label]
                for landmark in hand_landmarks.landmark:
                    landmark_values.extend([landmark.x, landmark.y, landmark.z])
                writer.writerow(landmark_values)
                self.detected_hands += 1
                return True
            else:
                # Jika masih gagal setelah augmentasi, pindahkan ke folder error
                error_path = os.path.join(self.error_dir, os.path.basename(image_path))
                shutil.copy2(image_path, error_path)
                self.total_error_images += 1
                self.undetected_hands += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error memproses {image_path}: {str(e)}")
            self.undetected_hands += 1
            return False

    def process_dataset(self):
        """Memproses seluruh dataset dan mengekstrak fitur"""
        self.logger.info("Memulai proses ekstraksi fitur...")
        
        # Analisis dataset terlebih dahulu
        class_distribution = self.analyze_dataset()
        
        with open(self.output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Header untuk 21 keypoint dengan koordinat xyz
            header = ["label"] + [f"landmark_{i}_{coord}" 
                                for i in range(21) 
                                for coord in ['x', 'y', 'z']]
            writer.writerow(header)

            with self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            ) as hands:
                for class_name, count in class_distribution.items():
                    class_path = os.path.join(self.dataset_path, class_name)
                    self.logger.info(f"Memproses kelas: {class_name}")
                    
                    image_files = [f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    with tqdm(total=len(image_files), 
                            desc=f"Memproses {class_name}") as pbar:
                        for image_file in image_files:
                            image_path = os.path.join(class_path, image_file)
                            self._extract_landmarks(image_path, class_name, hands, writer)
                            pbar.update(1)

        # Laporan akhir
        self._generate_final_report()

    def _generate_final_report(self):
        """Menghasilkan laporan akhir proses"""
        detection_rate = (self.detected_hands / self.total_images) * 100
        augmentation_rate = (self.augmented_detections / self.detected_hands) * 100 if self.detected_hands > 0 else 0
        error_rate = (self.total_error_images / self.total_images) * 100

        self.logger.info("\n=== Laporan Akhir Pemrosesan Dataset ===")
        self.logger.info(f"Total gambar diproses: {self.total_images}")
        self.logger.info(f"Landmark terdeteksi: {self.detected_hands} ({detection_rate:.2f}%)")
        self.logger.info(f"Terdeteksi setelah augmentasi: {self.augmented_detections} ({augmentation_rate:.2f}%)")
        self.logger.info(f"Gagal terdeteksi: {self.undetected_hands}")
        self.logger.info(f"Gambar error: {self.total_error_images} ({error_rate:.2f}%)")
        self.logger.info("=====================================")

if __name__ == "__main__":
    DATASET_PATH = "./data/raw"
    OUTPUT_FILE = "./output/features/hand_landmark_features.csv"
    
    processor = HandLandmarkProcessor(DATASET_PATH, OUTPUT_FILE)
    processor.process_dataset()