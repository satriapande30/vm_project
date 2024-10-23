import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
import logging
from datetime import datetime
import os
from collections import deque

class VirtualMouseSystem:
    def setup_logging(self):
        """Setup basic logging"""
        try:
            log_dir = 'output/log'
            os.makedirs(log_dir, exist_ok=True)
            
            self.logger = logging.getLogger('VirtualMouse')
            self.logger.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # File handler
            fh = logging.FileHandler(f'{log_dir}/virtual_mouse_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

    def load_model(self):
        """Load SVM model"""
        try:
            model_path = 'output/model/1svm_model.pkl'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.svm_model = model_data['model']
            self.scaler = model_data['scaler']
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def __init__(self):
        try:
            # Inisialisasi atribut dasar
            self.prev_hand_center = None
            self.pose_buffer = deque(maxlen=5)
            self.smoothing_factor = 0.5
            
            # Flank detection variables
            self.pose_states = {
                'v_pose': False,
                'middle_finger': False,
                'index_finger': False,
                'fist': False,
                'palm': False
            }
            self.prev_pose = None
            self.current_pose = None
            
            # Setup logging
            self.setup_logging()
            self.logger.info("Inisialisasi Virtual Mouse System...")
            
            # Load model SVM
            self.load_model()
            
            # Setup MediaPipe
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            # Setup PyAutoGUI
            pyautogui.FAILSAFE = False
            self.screen_width, self.screen_height = pyautogui.size()
            
            self.logger.info("Inisialisasi selesai")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def calculate_finger_distances(self, hand_landmarks):
        """Calculate Euclidean distances for classification"""
        try:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            wrist = landmarks[0]
            
            finger_tips = {
                'thumb': 4,
                'index': 8,
                'middle': 12,
                'ring': 16,
                'pinky': 20
            }
            
            distances = []
            
            # Distance from wrist to fingertips
            for tip_idx in finger_tips.values():
                tip = landmarks[tip_idx]
                distances.append(np.linalg.norm(tip - wrist))
            
            # Distances between fingertips
            for i, idx1 in enumerate(finger_tips.values()):
                for idx2 in list(finger_tips.values())[i+1:]:
                    tip1 = landmarks[idx1]
                    tip2 = landmarks[idx2]
                    distances.append(np.linalg.norm(tip1 - tip2))
            
            return distances
            
        except Exception as e:
            self.logger.error(f"Error calculating finger distances: {str(e)}")
            return None

    def predict_pose(self, features):
        """Predict hand pose"""
        try:
            if features is None:
                return None
            features_scaled = self.scaler.transform([features])
            return self.svm_model.predict(features_scaled)[0]
        except Exception as e:
            self.logger.error(f"Error predicting pose: {str(e)}")
            return None

    def smooth_prediction(self, pose):
        """Smooth predictions"""
        try:
            if pose is None:
                return None
            self.pose_buffer.append(pose)
            return max(set(self.pose_buffer), key=self.pose_buffer.count)
        except Exception as e:
            self.logger.error(f"Error smoothing prediction: {str(e)}")
            return None

    def detect_flank(self, pose):
        """
        Deteksi flank (perubahan status) dari pose
        Returns:
        - rising_edge: True jika pose baru terdeteksi
        - falling_edge: True jika pose berhenti terdeteksi
        """
        rising_edge = False
        falling_edge = False
        
        # Update pose saat ini
        self.prev_pose = self.current_pose
        self.current_pose = pose
        
        # Deteksi rising edge (pose baru terdeteksi)
        if self.prev_pose != pose and pose is not None:
            rising_edge = True
            self.pose_states[pose] = True
        
        # Deteksi falling edge (pose berhenti terdeteksi)
        if self.prev_pose is not None and pose != self.prev_pose:
            falling_edge = True
            if self.prev_pose in self.pose_states:
                self.pose_states[self.prev_pose] = False
        
        return rising_edge, falling_edge

    def execute_mouse_action(self, pose, hand_center):
        """Execute mouse actions dengan flank detection"""
        try:
            if hand_center is None or pose is None:
                return

            # Deteksi flank untuk pose saat ini
            rising_edge, falling_edge = self.detect_flank(pose)

            # Smooth movement untuk tracking posisi
            if self.prev_hand_center is not None:
                smooth_x = int((1 - self.smoothing_factor) * self.prev_hand_center[0] + 
                             self.smoothing_factor * hand_center[0])
                smooth_y = int((1 - self.smoothing_factor) * self.prev_hand_center[1] + 
                             self.smoothing_factor * hand_center[1])
                hand_center = (smooth_x, smooth_y)
            
            self.prev_hand_center = hand_center

            # Map coordinates
            screen_x = np.interp(hand_center[0], [100, 540], [0, self.screen_width])
            screen_y = np.interp(hand_center[1], [100, 380], [0, self.screen_height])
            
            # Execute actions based on flank detection
            if pose == 'v_pose':
                if rising_edge:  # Hanya aktifkan mode tracking saat pose pertama terdeteksi
                    self.logger.info("Tracking mode activated")
                if self.pose_states['v_pose']:  # Tracking hanya aktif selama pose dipertahankan
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
            
            elif pose == 'middle_finger' and rising_edge:
                self.logger.info("Left click executed")
                pyautogui.click()
            
            elif pose == 'index_finger' and rising_edge:
                self.logger.info("Right click executed")
                pyautogui.rightClick()
            
            elif pose == 'fist':
                if rising_edge:  # Mulai drag hanya pada rising edge
                    self.logger.info("Drag started")
                    pyautogui.mouseDown()
                if self.pose_states['fist']:  # Update posisi selama drag
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
            
            elif pose == 'palm':
                if rising_edge:  # Lepas drag hanya pada rising edge
                    self.logger.info("Drag released")
                    pyautogui.mouseUp()
            
            # Reset state jika pose berubah
            if falling_edge:
                self.logger.info(f"Pose {self.prev_pose} deactivated")
                
        except Exception as e:
            self.logger.error(f"Error executing mouse action: {str(e)}")

    def run(self):
        """Run the virtual mouse system"""
        try:
            self.logger.info("Starting virtual mouse system...")
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to grab frame")
                    break

                # Process frame
                image = cv2.flip(frame, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_image)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Process hand
                        features = self.calculate_finger_distances(hand_landmarks)
                        pose = self.predict_pose(features)
                        smooth_pose = self.smooth_prediction(pose)
                        
                        # Get hand center
                        hand_center = (
                            int(hand_landmarks.landmark[9].x * image.shape[1]),
                            int(hand_landmarks.landmark[9].y * image.shape[0])
                        )
                        
                        # Execute action
                        self.execute_mouse_action(smooth_pose, hand_center)
                        
                        # Display pose and status
                        status_text = f"Pose: {smooth_pose}"
                        if smooth_pose in self.pose_states and self.pose_states[smooth_pose]:
                            status_text += " (Active)"
                        
                        cv2.putText(
                            image,
                            status_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                # Show frame
                cv2.imshow('Virtual Mouse', image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("System stopped")

if __name__ == "__main__":
    try:
        virtual_mouse = VirtualMouseSystem()
        virtual_mouse.run()
    except Exception as e:
        print(f"Critical error: {str(e)}")