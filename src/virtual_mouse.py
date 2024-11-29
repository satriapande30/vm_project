import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
import logging
from datetime import datetime
import os
from collections import deque
import time

class VirtualMouseSystem:
    def setup_logging(self):
        """Setup basic logging"""
        try:
            log_dir = 'output/log'
            os.makedirs(log_dir, exist_ok=True)
            
            self.logger = logging.getLogger('VirtualMouse')
            self.logger.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            fh = logging.FileHandler(f'{log_dir}/virtual_mouse_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

    def load_model(self):
        """Load SVM model"""
        try:
            model_path = 'output/model/new_svm_model.pkl'
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
            # Initialize pose classes
            self.pose_classes = ['fist', 'palm', 'index_finger', 'middle_finger', 
                               'v_pose', 'v_pose_closed', 'random']
            
            self.prev_hand_center = None
            self.pose_buffer = deque(maxlen=5)
            self.smoothing_factor = 0.5
            
            # Boundary coordinates
            self.boundary_left = 100
            self.boundary_right = 540
            self.boundary_top = 100
            self.boundary_bottom = 380
            
            # Initialize pose states
            self.pose_states = {pose: False for pose in self.pose_classes}
            self.prev_pose = None
            self.current_pose = None
            
            # Colors for visualization (BGR format)
            self.colors = {
                'v_pose': (0, 255, 0),        # Green
                'v_pose_closed': (255, 0, 255),# Magenta
                'middle_finger': (0, 0, 255),  # Red
                'index_finger': (255, 165, 0), # Orange
                'fist': (255, 255, 0),        # Cyan
                'palm': (128, 0, 255),        # Purple
                'random': (169, 169, 169),     # Gray
                'boundary': (0, 140, 255)      # Dark Orange
            }
            
            # Setup logging and model
            self.setup_logging()
            self.logger.info("Initializing Virtual Mouse System...")
            self.load_model()
            
            # MediaPipe initialization
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            # Drawing specifications
            self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=2,
                circle_radius=2
            )
            self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
                color=(255, 0, 0),
                thickness=2
            )
            
            # PyAutoGUI setup
            pyautogui.FAILSAFE = False
            self.screen_width, self.screen_height = pyautogui.size()
            
            self.logger.info("Initialization complete")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def calculate_finger_distances(self, hand_landmarks):
        """Calculate Euclidean distances based on the training model's feature extraction"""
        try:
            distances = {}
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # 1. Get wrist as reference point
            wrist = landmarks[0]
            
            # 2. Calculate distances for each finger segment
            finger_landmarks = {
                'thumb': [1, 2, 3, 4],
                'index': [5, 6, 7, 8],
                'middle': [9, 10, 11, 12],
                'ring': [13, 14, 15, 16],
                'pinky': [17, 18, 19, 20]
            }
            
            # Calculate distances from wrist to each finger segment
            for finger, points in finger_landmarks.items():
                for i, lm_idx in enumerate(points):
                    point = landmarks[lm_idx]
                    distances[f'{finger}_segment_{i+1}'] = np.linalg.norm(point - wrist)
            
            # 3. Calculate distances between fingertips
            finger_tips = {
                'thumb': 4,
                'index': 8,
                'middle': 12,
                'ring': 16,
                'pinky': 20
            }
            
            for i, (f1, idx1) in enumerate(finger_tips.items()):
                for f2, idx2 in list(finger_tips.items())[i+1:]:
                    tip1 = landmarks[idx1]
                    tip2 = landmarks[idx2]
                    distances[f'{f1}_to_{f2}'] = np.linalg.norm(tip1 - tip2)
            
            # 4. Calculate flexion (distance from fingertip to MCP)
            mcp_points = {'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17}
            for finger, tip_idx in finger_tips.items():
                if finger != 'thumb':
                    tip = landmarks[tip_idx]
                    mcp = landmarks[mcp_points[finger]]
                    distances[f'{finger}_flexion'] = np.linalg.norm(tip - mcp)
            
            return list(distances.values())
            
        except Exception as e:
            self.logger.error(f"Error calculating finger distances: {str(e)}")
            return None

    def predict_pose(self, features):
        """Predict hand pose using the trained model"""
        try:
            if features is None:
                return None
            features_scaled = self.scaler.transform([features])
            prediction = self.svm_model.predict(features_scaled)[0]
            probability = self.svm_model.predict_proba(features_scaled)[0]
            max_prob = max(probability)
            
            # Only return prediction if confidence is high enough
            if max_prob > 0.7:  # Threshold for confidence
                return prediction
            return None
        except Exception as e:
            self.logger.error(f"Error predicting pose: {str(e)}")
            return None

    def draw_boundary_frame(self, image, hand_landmarks=None):
        """Draw boundary frame to indicate tracking limits"""
        # Draw main boundary rectangle
        cv2.rectangle(
            image,
            (self.boundary_left, self.boundary_top),
            (self.boundary_right, self.boundary_bottom),
            self.colors['boundary'],
            2
        )
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        padding = 10
        
        # Add "Tracking Area" label
        cv2.putText(
            image,
            "Tracking Area",
            (self.boundary_left, self.boundary_top - padding),
            font,
            font_scale,
            self.colors['boundary'],
            font_thickness
        )
        
        # Add warning text if hand is outside boundaries
        if hand_landmarks:
            hand_x = int(hand_landmarks.landmark[9].x * image.shape[1])
            hand_y = int(hand_landmarks.landmark[9].y * image.shape[0])
            
            if (hand_x < self.boundary_left or hand_x > self.boundary_right or
                hand_y < self.boundary_top or hand_y > self.boundary_bottom):
                cv2.putText(
                    image,
                    "Warning: Hand outside tracking area!",
                    (10, image.shape[0] - 20),
                    font,
                    font_scale,
                    (0, 0, 255),  # Red color for warning
                    font_thickness
                )

    def draw_bounding_box(self, image, hand_landmarks, pose):
        """Draw bounding box around hand"""
        h, w, _ = image.shape
        x_coords = []
        y_coords = []
        
        # Get landmark coordinates
        for landmark in hand_landmarks.landmark:
            x_coords.append(int(landmark.x * w))
            y_coords.append(int(landmark.y * h))
        
        # Calculate bounding box with padding
        padding = 20
        x1 = max(0, min(x_coords) - padding)
        y1 = max(0, min(y_coords) - padding)
        x2 = min(w, max(x_coords) + padding)
        y2 = min(h, max(y_coords) + padding)
        
        # Get color based on pose
        color = self.colors.get(pose, self.colors['random']) if pose else self.colors['random']
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add Pose label
        if pose:
            label = f"Pose: {pose}"
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        return x1, y1, x2, y2

    def run(self):
        pTime = 0
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

                # Draw boundary frame
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    self.draw_boundary_frame(image, hand_landmarks)
                else:
                    self.draw_boundary_frame(image)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    features = self.calculate_finger_distances(hand_landmarks)
                    pose = self.predict_pose(features)
                    
                    if pose:
                        self.pose_buffer.append(pose)
                        smooth_pose = max(set(self.pose_buffer), key=self.pose_buffer.count)
                    else:
                        smooth_pose = None

                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.landmark_drawing_spec,
                        self.connection_drawing_spec
                    )

                    if smooth_pose:
                        self.draw_bounding_box(image, hand_landmarks, smooth_pose)

                    if smooth_pose:
                        hand_center = (
                            int(hand_landmarks.landmark[9].x * image.shape[1]),
                            int(hand_landmarks.landmark[9].y * image.shape[0])
                        )
                        self.execute_mouse_action(smooth_pose, hand_center)

                # Calculate and display FPS
                cTime = time.time()
                self.fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(
                    image,
                    f'FPS: {int(self.fps)}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow('Virtual Mouse', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("System stopped")

    def execute_mouse_action(self, pose, hand_center):
        """Execute mouse actions based on detected pose"""
        try:
            if hand_center is None or pose is None:
                return

            # Check boundaries
            x, y = hand_center
            if (x < self.boundary_left or x > self.boundary_right or
                y < self.boundary_top or y > self.boundary_bottom):
                return

            # Smooth hand movement
            if self.prev_hand_center is not None:
                x = int((1 - self.smoothing_factor) * self.prev_hand_center[0] + 
                       self.smoothing_factor * x)
                y = int((1 - self.smoothing_factor) * self.prev_hand_center[1] + 
                       self.smoothing_factor * y)
            self.prev_hand_center = (x, y)

            # Map coordinates
            screen_x = np.interp(x, [self.boundary_left, self.boundary_right], 
                               [0, self.screen_width])
            screen_y = np.interp(y, [self.boundary_top, self.boundary_bottom], 
                               [0, self.screen_height])

            # Execute actions based on pose
            if pose == 'v_pose':
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)
                self.logger.info(f"Moving cursor to: ({screen_x}, {screen_y})")
            
            elif pose == 'v_pose_closed' and pose != self.prev_pose:
                pyautogui.doubleClick()
                self.logger.info("Double click executed")
            
            elif pose == 'middle_finger' and pose != self.prev_pose:
                pyautogui.click()
                self.logger.info("Left click executed")
            
            elif pose == 'index_finger' and pose != self.prev_pose:
                pyautogui.rightClick()
                self.logger.info("Right click executed")
            
            elif pose == 'fist':
                if pose != self.prev_pose:
                    pyautogui.mouseDown()
                    self.logger.info(f"Starting drag at: ({screen_x}, {screen_y})")
                else:
                    # Menampilkan koordinat saat proses drag
                    self.logger.info(f"Moving drag to: ({screen_x}, {screen_y})")
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)
            
            elif pose == 'palm' and pose != self.prev_pose:
                pyautogui.mouseUp()
                self.logger.info("Ending drag")

            self.prev_pose = pose

        except Exception as e:
            self.logger.error(f"Error executing mouse action: {str(e)}")

if __name__ == "__main__":
    try:
        virtual_mouse = VirtualMouseSystem()
        virtual_mouse.run()
    except Exception as e:
        print(f"Critical error: {str(e)}")
