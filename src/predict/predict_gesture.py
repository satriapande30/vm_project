import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging
import random
from tqdm import tqdm
import joblib
from datetime import datetime

class GesturePrediction:
    def __init__(self, model_path, output_dir):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        self.model_path = model_path
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        self.load_model()

    def _setup_logger(self):
        log_dir = os.path.join(self.output_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"predict_gesture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        return logger

    def load_model(self):
        self.logger.info(f"Loading model from {self.model_path}")
        try:
            self.model, self.scaler, self.label_encoder = joblib.load(self.model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def predict_gesture(self, landmarks):
        distances = []
        base_point = landmarks[0]
        for point in [4, 8, 12, 16, 20]:
            distance = self.euclidean_distance(base_point, landmarks[point])
            distances.append(distance)

        # Definisi gestur berdasarkan jarak Euclidean
        if all(d > 0.1 for d in distances):
            return "palm"
        elif all(d < 0.05 for d in distances):
            return "fist"
        elif distances[1] > 0.1 and all(d < 0.05 for i, d in enumerate(distances) if i != 1):
            return "index"
        elif distances[2] > 0.1 and all(d < 0.05 for i, d in enumerate(distances) if i != 2):
            return "mid"
        elif distances[1] > 0.1 and distances[2] > 0.1 and self.euclidean_distance(landmarks[8], landmarks[12]) > 0.05:
            return "v_gest"
        elif distances[1] > 0.1 and distances[2] > 0.1 and self.euclidean_distance(landmarks[8], landmarks[12]) < 0.05:
            return "two_finger_closed"
        else:
            return "unknown"

    def process_image(self, image_path):
        self.logger.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Predict using SVM model
            landmarks_flat = landmarks.flatten()
            landmarks_scaled = self.scaler.transform([landmarks_flat])
            svm_prediction = self.label_encoder.inverse_transform(self.model.predict(landmarks_scaled))[0]
            
            # Predict using Euclidean distance
            euclidean_prediction = self.predict_gesture(landmarks)

            # Menggambar landmark dan garis Euclidean distance
            self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            base_point = tuple(map(int, [landmarks[0][0] * image.shape[1], landmarks[0][1] * image.shape[0]]))
            for point in [4, 8, 12, 16, 20]:
                end_point = tuple(map(int, [landmarks[point][0] * image.shape[1], landmarks[point][1] * image.shape[0]]))
                cv2.line(image, base_point, end_point, (0, 255, 0), 2)

            return image, svm_prediction, euclidean_prediction
        else:
            return image, "No hand detected", "No hand detected"

    def run_prediction(self, input_dir):
        self.logger.info("Starting gesture prediction")
        try:
            predict_dir = os.path.join(self.output_dir, 'predict')
            os.makedirs(predict_dir, exist_ok=True)

            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Input directory not found: {input_dir}")

            label_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

            if not label_folders:
                self.logger.warning(f"No label folders found in {input_dir}")
                return

            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            axs = axs.ravel()

            for i, label in enumerate(tqdm(label_folders, desc="Processing gestures")):
                label_dir = os.path.join(input_dir, label)
                image_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not image_files:
                    self.logger.warning(f"No image files found in {label_dir}")
                    continue

                # Select one random image from each label folder
                sample_image = random.choice(image_files)
                image_path = os.path.join(label_dir, sample_image)
                
                output_image, svm_prediction, euclidean_prediction = self.process_image(image_path)
                
                # Display the image with landmarks and Euclidean distance lines
                axs[i].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                axs[i].set_title(f"True: {label}\nSVM: {svm_prediction}\nEuclidean: {euclidean_prediction}")
                axs[i].axis('off')

                self.logger.info(f"Processed {sample_image}: True - {label}, SVM - {svm_prediction}, Euclidean - {euclidean_prediction}")

            plt.tight_layout()

            # Find the next available file number
            count = 1
            while os.path.exists(os.path.join(predict_dir, f"gesture_predictions_{count}.jpg")):
                count += 1

            output_filename = f"gesture_predictions_{count}.jpg"
            plt.savefig(os.path.join(predict_dir, output_filename))
            plt.close()

            self.logger.info(f"Gesture prediction completed. Output saved as {output_filename}")
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        MODEL_PATH = "D:/KULIAH/SKRIPSI/vm_project/output/model/svm_model.pkl"
        INPUT_DIR = "D:/KULIAH/SKRIPSI/vm_project/test_data/"
        OUTPUT_DIR = "D:/KULIAH/SKRIPSI/vm_project/output"

        gesture_predictor = GesturePrediction(MODEL_PATH, OUTPUT_DIR)
        gesture_predictor.run_prediction(INPUT_DIR)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")