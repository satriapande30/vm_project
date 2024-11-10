#TRAINING DAN TESTING DATASET

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

class HandPoseTrainer:
    def __init__(self, input_file, batch_size=1000):
        self.input_file = input_file
        self.batch_size = batch_size
        self.setup_directories()
        self.logger = self._setup_logger()
        self.scaler = StandardScaler()
        # Define class names for hand poses
        self.pose_classes = ['fist', 'palm', 'index_finger', 'middle_finger', 'v_pose', 'v_pose_closed', 'random']
        
    def setup_directories(self):
        """Membuat direktori yang diperlukan"""
        directories = [
            'output/log',
            'output/model',
            'output/visualization'
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def _setup_logger(self):
        """Setup sistem logging"""
        log_file = f'output/log/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def calculate_finger_distances(self, data):
        """Menghitung jarak Euclidean untuk setiap jari"""
        distances = {}
        
        # Titik referensi landmark [0]
        wrist = np.array([
            data[f'landmark_0_x'],
            data[f'landmark_0_y'],
            data[f'landmark_0_z']
        ])
        
        # Titik ujung jari [4,8,12,16,20]
        finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        # Hitung jarak dari pergelangan ke ujung jari
        for finger, tip_idx in finger_tips.items():
            tip = np.array([
                data[f'landmark_{tip_idx}_x'],
                data[f'landmark_{tip_idx}_y'],
                data[f'landmark_{tip_idx}_z']
            ])
            distances[f'{finger}_to_wrist'] = np.linalg.norm(tip - wrist)
        
        # Hitung jarak antar ujung jari
        for i, (f1, idx1) in enumerate(finger_tips.items()):
            for f2, idx2 in list(finger_tips.items())[i+1:]:
                tip1 = np.array([
                    data[f'landmark_{idx1}_x'],
                    data[f'landmark_{idx1}_y'],
                    data[f'landmark_{idx1}_z']
                ])
                tip2 = np.array([
                    data[f'landmark_{idx2}_x'],
                    data[f'landmark_{idx2}_y'],
                    data[f'landmark_{idx2}_z']
                ])
                distances[f'{f1}_to_{f2}'] = np.linalg.norm(tip1 - tip2)
        
        return distances

    def process_batch(self, batch_df):
        """Memproses satu batch data"""
        features = []
        for _, row in batch_df.iterrows():
            distances = self.calculate_finger_distances(row)
            features.append(list(distances.values()))
        return features

    def prepare_data(self):
        """Menyiapkan data untuk training"""
        self.logger.info("Membaca dan menyiapkan dataset...")
        
        df = pd.read_csv(self.input_file)
        total_samples = len(df)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        features = []
        labels = df['label'].values
        
        for i in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_samples)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Parallel processing untuk batch
            batch_features = Parallel(n_jobs=-1)(
                delayed(self.calculate_finger_distances)(row)
                for _, row in batch_df.iterrows()
            )
            features.extend([list(d.values()) for d in batch_features])
        
        features = np.array(features)
        
        # Normalisasi fitur
        features = self.scaler.fit_transform(features)
        
        return features, labels

    def train_model(self):
        """Melatih model SVM"""
        self.logger.info("Memulai proses training...")
        
        # Persiapan data
        X, y = self.prepare_data()
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Training SVM
        self.logger.info("Training model SVM...")
        svm = SVC(kernel='rbf', probability=True)
        svm.fit(X_train, y_train)
        
        # Evaluasi model
        train_score = svm.score(X_train, y_train)
        test_score = svm.score(X_test, y_test)
        
        self.logger.info(f"Training accuracy: {train_score:.4f}")
        self.logger.info(f"Testing accuracy: {test_score:.4f}")
        
        # Prediksi dan evaluasi
        y_pred = svm.predict(X_test)
        
        # Simpan model
        model_path = 'output/model/1svm_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({'model': svm, 'scaler': self.scaler}, f)
        
        # Visualisasi hasil
        self.visualize_results(y_test, y_pred, train_score, test_score)
        
        return svm, (X_train, X_test, y_train, y_test)

    def visualize_results(self, y_true, y_pred, train_score, test_score):
        """Membuat visualisasi hasil training yang lebih komprehensif"""
        self.logger.info("Membuat visualisasi hasil...")
        
        # 1. Enhanced Confusion Matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix heatmap with class names
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=self.pose_classes,
                   yticklabels=self.pose_classes)
        plt.title('Confusion Matrix - Hand Pose Classification', pad=20)
        plt.xlabel('Predicted Pose')
        plt.ylabel('True Pose')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig('output/visualization/1confusion_matrix.png')
        plt.close()
        
        # 2. Detailed Classification Report
        report = classification_report(y_true, y_pred, 
                                    target_names=self.pose_classes,
                                    output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(report_df.iloc[:-3, :3].astype(float), 
                    annot=True, 
                    cmap='YlOrRd',
                    xticklabels=['Precision', 'Recall', 'F1-Score'],
                    yticklabels=self.pose_classes)
        plt.title('Classification Metrics by Class')
        plt.tight_layout()
        plt.savefig('output/visualization/1classification_report.png')
        plt.close()
        
        # 3. Overall Model Performance Metrics
        plt.figure(figsize=(10, 6))
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1-Score': f1_score(y_true, y_pred, average='weighted')
        }
        
        bars = plt.bar(metrics.keys(), metrics.values())
        plt.title('Overall Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('output/visualization/1overall_metrics.png')
        plt.close()
        
        # 4. Training vs Testing Performance Comparison
        plt.figure(figsize=(10, 6))
        performance_data = {
            'Training': train_score,
            'Testing': test_score
        }
        
        bars = plt.bar(performance_data.keys(), performance_data.values(),
                      color=['#2ecc71', '#3498db'])
        plt.title('Training vs Testing Accuracy')
        plt.ylabel('Accuracy Score')
        plt.ylim(0, 1.0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('output/visualization/1model_performance.png')
        plt.close()
        
        # 5. Per-Class Performance Metrics
        plt.figure(figsize=(15, 8))
        
        # Calculate per-class metrics
        class_precision = precision_score(y_true, y_pred, average=None)
        class_recall = recall_score(y_true, y_pred, average=None)
        class_f1 = f1_score(y_true, y_pred, average=None)
        
        x = np.arange(len(self.pose_classes))
        width = 0.25
        
        plt.bar(x - width, class_precision, width, label='Precision')
        plt.bar(x, class_recall, width, label='Recall')
        plt.bar(x + width, class_f1, width, label='F1-Score')
        
        plt.xlabel('Hand Pose Classes')
        plt.ylabel('Score')
        plt.title('Performance Metrics per Class')
        plt.xticks(x, self.pose_classes, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig('output/visualization/1per_class_metrics.png')
        plt.close()

if __name__ == "__main__":
    INPUT_FILE = "output/features/hand_landmark_features.csv"
    
    trainer = HandPoseTrainer(INPUT_FILE)
    model, (X_train, X_test, y_train, y_test) = trainer.train_model()
