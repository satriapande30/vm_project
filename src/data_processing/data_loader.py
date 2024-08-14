import os
import logging
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import shutil

class DataLoader:
    def __init__(self, dataset_path, batch_size=1000):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.logger = self._setup_logger()
        self.error_image_dir = 'output/error_image'
        if not os.path.exists(self.error_image_dir):
            os.makedirs(self.error_image_dir)

    def _setup_logger(self):
        log_dir = 'output/log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"data_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logger = logging.getLogger('DataLoader')
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        return logger

    def count_images(self):
        total_count = 0
        class_counts = {}
        anomaly_count = 0

        self.logger.info("Mulai menghitung dan memeriksa data citra...")

        for class_folder in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_folder)
            if os.path.isdir(class_path):
                class_count = 0
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for i in tqdm(range(0, len(image_files), self.batch_size), desc=f"Memuat Dataset {class_folder}"):
                    batch = image_files[i:i+self.batch_size]
                    for image_file in batch:
                        image_path = os.path.join(class_path, image_file)
                        try:
                            with Image.open(image_path) as img:
                                if img.size == (640, 480):
                                    class_count += 1
                                else:
                                    self.logger.warning(f"Ukuran gambar tidak sesuai: {image_path}")
                                    anomaly_count += 1
                                    self._move_anomaly_image(image_path, class_folder)
                        except Exception as e:
                            self.logger.error(f"Error membaca gambar {image_path}: {str(e)}")
                            anomaly_count += 1
                            self._move_anomaly_image(image_path, class_folder)
                
                class_counts[class_folder] = class_count
                total_count += class_count

        self.logger.info("Selesai menghitung dan memeriksa data citra.")
        self.logger.info(f"Total gambar valid: {total_count}")
        self.logger.info(f"Total gambar anomali: {anomaly_count}")
        for class_name, count in class_counts.items():
            self.logger.info(f"Kelas {class_name}: {count} gambar valid")

        return total_count, class_counts, anomaly_count

    def _move_anomaly_image(self, image_path, class_folder):
        filename = os.path.basename(image_path)
        error_class_dir = os.path.join(self.error_image_dir, class_folder)
        if not os.path.exists(error_class_dir):
            os.makedirs(error_class_dir)
        
        new_path = os.path.join(error_class_dir, filename)
        shutil.move(image_path, new_path)
        self.logger.info(f"Gambar anomali dipindahkan ke: {new_path}")

if __name__ == "__main__":
    DATASET_PATH = "D:/KULIAH/SKRIPSI/vm_project/data/"
    loader = DataLoader(DATASET_PATH)
    total, class_counts, anomaly_count = loader.count_images()