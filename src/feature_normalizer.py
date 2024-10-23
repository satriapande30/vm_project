import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import logging
from datetime import datetime

class HandFeatureNormalizer:
    def __init__(self, input_file, output_dir='output/features'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        
        # Memastikan direktori output ada
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _setup_logger(self):
        """Setup sistem logging"""
        log_dir = 'output/log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"feature_normalizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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

    def calculate_euclidean_distance(self, row, point1, point2):
        """
        Menghitung Euclidean distance antara dua titik landmark
        point1, point2: indeks landmark (0-20)
        """
        p1_x = row[f'landmark_{point1}_x']
        p1_y = row[f'landmark_{point1}_y']
        p1_z = row[f'landmark_{point1}_z']
        
        p2_x = row[f'landmark_{point2}_x']
        p2_y = row[f'landmark_{point2}_y']
        p2_z = row[f'landmark_{point2}_z']
        
        return np.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2 + (p2_z - p1_z)**2)

    def normalize_features(self):
        """
        Melakukan normalisasi fitur menggunakan Euclidean distance
        antara landmark 0 (pergelangan tangan) dan 17 (pangkal jari tengah)
        """
        self.logger.info("Memulai proses normalisasi fitur...")
        
        try:
            # Membaca dataset
            self.logger.info("Membaca file dataset...")
            df = pd.read_csv(self.input_file)
            total_rows = len(df)
            self.logger.info(f"Total data yang akan dinormalisasi: {total_rows}")

            # Menghitung distance reference (antara landmark 0 dan 17)
            self.logger.info("Menghitung distance reference...")
            df['reference_distance'] = df.apply(
                lambda row: self.calculate_euclidean_distance(row, 0, 17), 
                axis=1
            )

            # Normalisasi setiap koordinat landmark
            normalized_data = []
            self.logger.info("Melakukan normalisasi koordinat...")
            
            for index, row in tqdm(df.iterrows(), total=total_rows, desc="Normalizing features"):
                normalized_row = {'label': row['label']}
                ref_distance = row['reference_distance']
                
                # Normalisasi setiap landmark
                for i in range(21):  # 21 landmark points
                    for coord in ['x', 'y', 'z']:
                        orig_value = row[f'landmark_{i}_{coord}']
                        # Normalisasi relatif terhadap landmark 0
                        base_value = row[f'landmark_0_{coord}']
                        normalized_value = (orig_value - base_value) / ref_distance
                        normalized_row[f'landmark_{i}_{coord}'] = normalized_value
                
                normalized_data.append(normalized_row)

            # Membuat DataFrame dari hasil normalisasi
            normalized_df = pd.DataFrame(normalized_data)
            
            # Menyimpan hasil normalisasi
            output_file = os.path.join(self.output_dir, 'normalized_features.csv')
            normalized_df.to_csv(output_file, index=False)
            
            # Membuat file statistik
            self._generate_statistics(normalized_df, df['reference_distance'])
            
            self.logger.info(f"Normalisasi selesai. File disimpan di: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error dalam proses normalisasi: {str(e)}")
            raise

    def _generate_statistics(self, normalized_df, reference_distances):
        """Menghasilkan statistik dari hasil normalisasi"""
        stats_file = os.path.join(self.output_dir, 'normalization_statistics.txt')
        
        with open(stats_file, 'w') as f:
            f.write("=== Statistik Normalisasi ===\n\n")
            
            # Statistik reference distance
            f.write("Statistik Reference Distance (Landmark 0 ke 17):\n")
            f.write(f"Mean: {reference_distances.mean():.6f}\n")
            f.write(f"Std: {reference_distances.std():.6f}\n")
            f.write(f"Min: {reference_distances.min():.6f}\n")
            f.write(f"Max: {reference_distances.max():.6f}\n\n")
            
            # Statistik per kelas
            f.write("Distribusi Data per Kelas:\n")
            class_dist = normalized_df['label'].value_counts()
            for label, count in class_dist.items():
                f.write(f"{label}: {count} samples\n")
            
            # Statistik koordinat ternormalisasi
            f.write("\nRange Koordinat Ternormalisasi:\n")
            for coord in ['x', 'y', 'z']:
                coord_columns = [col for col in normalized_df.columns if col.endswith(f"_{coord}")]
                coord_min = normalized_df[coord_columns].min().min()
                coord_max = normalized_df[coord_columns].max().max()
                f.write(f"{coord}: [{coord_min:.6f}, {coord_max:.6f}]\n")

if __name__ == "__main__":
    INPUT_FILE = "./output/features/hand_landmark_features.csv"
    
    normalizer = HandFeatureNormalizer(INPUT_FILE)
    normalizer.normalize_features()