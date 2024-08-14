import sys
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import logging
from datetime import datetime
import os
import joblib

# Tambahkan root directory proyek dan direktori src ke sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from mouse_control import LANDMARK_POINTS, GESTURES


class VirtualMouse:
    def __init__(self, output_dir, model_path):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothening = 5
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        self.model_path = model_path
        self.load_model()
        self.is_dragging = False

    def _setup_logger(self):
        log_dir = os.path.join(self.output_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"virtual_mouse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
        # SVM prediction
        landmarks_flat = landmarks.flatten()
        landmarks_scaled = self.scaler.transform([landmarks_flat])
        svm_prediction = self.label_encoder.inverse_transform(self.model.predict(landmarks_scaled))[0]

        # Euclidean distance prediction
        distances = []
        base_point = landmarks[0]
        for point in [4, 8, 12, 16, 20]:
            distance = self.euclidean_distance(base_point, landmarks[point])
            distances.append(distance)

        if all(d > 0.1 for d in distances):
            euclidean_prediction = "palm"
        elif all(d < 0.05 for d in distances):
            euclidean_prediction = "fist"
        elif distances[1] > 0.1 and all(d < 0.05 for i, d in enumerate(distances) if i != 1):
            euclidean_prediction = "index"
        elif distances[2] > 0.1 and all(d < 0.05 for i, d in enumerate(distances) if i != 2):
            euclidean_prediction = "mid"
        elif distances[1] > 0.1 and distances[2] > 0.1 and self.euclidean_distance(landmarks[8], landmarks[12]) > 0.05:
            euclidean_prediction = "v_gest"
        elif distances[1] > 0.1 and distances[2] > 0.1 and self.euclidean_distance(landmarks[8], landmarks[12]) < 0.05:
            euclidean_prediction = "two_finger_closed"
        else:
            euclidean_prediction = "unknown"

        # Combine predictions (you can adjust this logic as needed)
        if svm_prediction == euclidean_prediction:
            return svm_prediction
        else:
            # In case of disagreement, you might want to prefer one method over the other
            # or implement some kind of voting system
            return svm_prediction  # or euclidean_prediction, depending on your preference

    def execute_mouse_action(self, gesture, landmarks):
        if gesture == GESTURES['FIST']:
            if not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
                self.logger.info("Started dragging (left click hold)")
            self.move_cursor(landmarks)
        elif gesture == GESTURES['PALM']:
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
                self.logger.info("Stopped dragging")
        elif gesture == GESTURES['INDEX']:
            pyautogui.click()  # Left click
            self.logger.info("Left click")
        elif gesture == GESTURES['MID']:
            pyautogui.rightClick()  # Right click
            self.logger.info("Right click")
        elif gesture == GESTURES['TWO_FINGER_CLOSED']:
            pyautogui.doubleClick()
            self.logger.info("Double click")
        elif gesture == GESTURES['V_GEST']:
            self.move_cursor(landmarks)

    def move_cursor(self, landmarks):
        index_finger_tip = landmarks[LANDMARK_POINTS['INDEX_FINGER_TIP']]
        x = int(index_finger_tip[0] * self.screen_width)
        y = int(index_finger_tip[1] * self.screen_height)

        self.curr_x = self.prev_x + (x - self.prev_x) / self.smoothening
        self.curr_y = self.prev_y + (y - self.prev_y) / self.smoothening

        pyautogui.moveTo(self.curr_x, self.curr_y)
        self.prev_x, self.prev_y = self.curr_x, self.curr_y
        
        self.logger.info(f"Moving cursor to ({self.curr_x}, {self.curr_y})")

    def run(self):
        cap = cv2.VideoCapture(0)
        self.logger.info("Virtual Mouse started")

        while True:
            success, image = cap.read()
            if not success:
                self.logger.error("Failed to capture frame")
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    gesture = self.predict_gesture(landmarks)

                    self.execute_mouse_action(gesture, landmarks)

                    # Display the detected gesture on the image
                    cv2.putText(image, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if self.is_dragging:
                        cv2.putText(image, "Dragging", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Virtual Mouse", image)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

        cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Virtual Mouse stopped")

if __name__ == "__main__":
    OUTPUT_DIR = "D:/KULIAH/SKRIPSI/vm_project/output"
    MODEL_PATH = "D:/KULIAH/SKRIPSI/vm_project/output/model/svm_model.pkl"
    virtual_mouse = VirtualMouse(OUTPUT_DIR, MODEL_PATH)
    virtual_mouse.run()