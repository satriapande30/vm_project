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
            # Existing initialization code...
            self.prev_hand_center = None
            self.pose_buffer = deque(maxlen=5)
            self.smoothing_factor = 0.5
            
            # Add boundary coordinates
            self.boundary_left = 100
            self.boundary_right = 540
            self.boundary_top = 100
            self.boundary_bottom = 380
            
            self.pose_states = {
                'v_pose': False,
                'middle_finger': False,
                'index_finger': False,
                'fist': False,
                'palm': False
            }
            self.prev_pose = None
            self.current_pose = None
            
            # Colors for visualization (BGR format)
            self.colors = {
                'v_pose': (0, 255, 0),      # Green
                'middle_finger': (0, 0, 255), # Red
                'index_finger': (255, 0, 0),  # Blue
                'fist': (0, 255, 255),       # Yellow
                'palm': (255, 0, 255),       # Magenta
                'default': (128, 128, 128),   # Gray
                'boundary': (255, 165, 0)     # Orange for boundary
            }
            
            # Setup logging
            self.setup_logging()
            self.logger.info("Initializing Virtual Mouse System...")
            
            # Rest of the initialization code...
            self.load_model()
            
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=2,
                circle_radius=2
            )
            self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
                color=(255, 0, 0),
                thickness=2
            )
            
            pyautogui.FAILSAFE = False
            self.screen_width, self.screen_height = pyautogui.size()
            
            self.logger.info("Initialization complete")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def draw_hand_landmarks(self, image, hand_landmarks):
        """Draw hand landmarks with custom drawing specs"""
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.landmark_drawing_spec,
            self.connection_drawing_spec
        )

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
        color = self.colors.get(pose, self.colors['default']) if pose else self.colors['default']
        
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
            
            for tip_idx in finger_tips.values():
                tip = landmarks[tip_idx]
                distances.append(np.linalg.norm(tip - wrist))
            
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
        """Detect flank changes in pose"""
        rising_edge = False
        falling_edge = False
        
        self.prev_pose = self.current_pose
        self.current_pose = pose
        
        if self.prev_pose != pose and pose is not None:
            rising_edge = True
            self.pose_states[pose] = True
        
        if self.prev_pose is not None and pose != self.prev_pose:
            falling_edge = True
            if self.prev_pose in self.pose_states:
                self.pose_states[self.prev_pose] = False
        
        return rising_edge, falling_edge

    def execute_mouse_action(self, pose, hand_center):
        """Execute mouse actions with boundary checking"""
        try:
            if hand_center is None or pose is None:
                return

            # Check if hand is within boundaries
            x, y = hand_center
            if (x < self.boundary_left or x > self.boundary_right or
                y < self.boundary_top or y > self.boundary_bottom):
                return  # Don't execute actions if hand is outside boundaries

            rising_edge, falling_edge = self.detect_flank(pose)

            if self.prev_hand_center is not None:
                smooth_x = int((1 - self.smoothing_factor) * self.prev_hand_center[0] + 
                             self.smoothing_factor * hand_center[0])
                smooth_y = int((1 - self.smoothing_factor) * self.prev_hand_center[1] + 
                             self.smoothing_factor * hand_center[1])
                hand_center = (smooth_x, smooth_y)
            
            self.prev_hand_center = hand_center

            # Map coordinates within boundary to screen coordinates
            screen_x = np.interp(hand_center[0], 
                               [self.boundary_left, self.boundary_right], 
                               [0, self.screen_width])
            screen_y = np.interp(hand_center[1], 
                               [self.boundary_top, self.boundary_bottom], 
                               [0, self.screen_height])
            
            # Rest of the mouse action code remains the same...
            if pose == 'v_pose':
                if rising_edge:
                    self.logger.info("Tracking mode activated")
                if self.pose_states['v_pose']:
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
            
            elif pose == 'middle_finger' and rising_edge:
                self.logger.info("Left click executed")
                pyautogui.click()
            
            elif pose == 'index_finger' and rising_edge:
                self.logger.info("Right click executed")
                pyautogui.rightClick()
            
            elif pose == 'fist':
                if rising_edge:
                    self.logger.info("Drag started")
                    pyautogui.mouseDown()
                if self.pose_states['fist']:
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
            
            elif pose == 'palm':
                if rising_edge:
                    self.logger.info("Drag released")
                    pyautogui.mouseUp()
            
            if falling_edge:
                self.logger.info(f"Pose {self.prev_pose} deactivated")
                
        except Exception as e:
            self.logger.error(f"Error executing mouse action: {str(e)}")

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

                # Draw boundary frame (initially without hand landmarks)
                self.draw_boundary_frame(image)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Update boundary frame with hand landmarks
                        self.draw_boundary_frame(image, hand_landmarks)
                        
                        # Get features and predict pose
                        features = self.calculate_finger_distances(hand_landmarks)
                        pose = self.predict_pose(features)
                        smooth_pose = self.smooth_prediction(pose)
                        
                        # Draw visualizations
                        self.draw_hand_landmarks(image, hand_landmarks)
                        x1, y1, x2, y2 = self.draw_bounding_box(image, hand_landmarks, smooth_pose)
                        
                        # Calculate hand center and execute action
                        hand_center = (
                            int(hand_landmarks.landmark[9].x * image.shape[1]),
                            int(hand_landmarks.landmark[9].y * image.shape[0])
                        )
                        self.execute_mouse_action(smooth_pose, hand_center)

                # Show FPS
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

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
