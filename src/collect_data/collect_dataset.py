import cv2
import os
import numpy as np
import time
import logging
from datetime import datetime

DATASET_PATH = 'vm_project/data/raw'
CAMERA_INDEX = 0
GESTURES = ['fist', 'palm', 'index', 'mid', 'v_gest', 'two_finger_closed']
HANDS = ['left', 'right']
DISTANCES = ['30cm', '60cm', '90cm']
IMAGE_SIZE = (640, 480)
IMAGE_EXTENSION = '.jpg'
LOG_DIR = 'output/log'

def setup_logging():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_file = os.path.join(LOG_DIR, f"collect_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def create_dataset_structure():
    for gesture in GESTURES:
        for hand in HANDS:
            for distance in DISTANCES:
                path = os.path.join(DATASET_PATH, gesture, hand, distance)
                if not os.path.exists(path):
                    os.makedirs(path)
                    logging.info(f"Created directory: {path}")

def capture_images():
    logging.info("Starting image capture process")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

    if not cap.isOpened():
        logging.error("Error: Unable to open camera.")
        return

    gesture_index = 0
    hand_index = 0
    distance_index = 0
    count = 0

    logging.info("Camera opened successfully")
    logging.info("Controls: 'g': change gesture, 'h': change hand, 'd': change distance, 'c': capture image, 'q': quit")

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Failed to capture image.")
            break

        frame = cv2.flip(frame, 1)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        gesture = GESTURES[gesture_index]
        hand = HANDS[hand_index]
        distance = DISTANCES[distance_index]

        avg_brightness = cv2.mean(frame)[0]
        lighting_condition_text = "Good" if avg_brightness > 100 else "Poor"

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Lighting: {lighting_condition_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Hand: {hand}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Pose: {gesture}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {distance}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Capture', frame)

        key = cv2.waitKey(1)
        if key == ord('g'):
            gesture_index = (gesture_index + 1) % len(GESTURES)
            logging.info(f"Changed gesture to: {GESTURES[gesture_index]}")
        elif key == ord('h'):
            hand_index = (hand_index + 1) % len(HANDS)
            logging.info(f"Changed hand to: {HANDS[hand_index]}")
        elif key == ord('d'):
            distance_index = (distance_index + 1) % len(DISTANCES)
            logging.info(f"Changed distance to: {DISTANCES[distance_index]}")
        elif key == ord('c'):
            img_name = f"{gesture}_{hand}_{distance}_{count}{IMAGE_EXTENSION}"
            img_path = os.path.join(DATASET_PATH, gesture, hand, distance, img_name)
            cv2.imwrite(img_path, frame)
            count += 1
            logging.info(f"Saved {img_path}")
            logging.info(f"Total images captured: {count}")
        elif key == ord('q'):
            logging.info(f"Quitting. Total images captured: {count}")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Image capture process completed")

def main():
    setup_logging()
    logging.info("Starting collect_dataset.py")
    create_dataset_structure()
    capture_images()
    logging.info("collect_dataset.py completed")

if __name__ == "__main__":
    main()