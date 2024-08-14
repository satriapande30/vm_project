import cv2
import mediapipe as mp
import math
import os
import random
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def euclidean_distance(landmark1, landmark2):
    x1, y1, z1 = landmark1.x, landmark1.y, landmark1.z
    x2, y2, z2 = landmark2.x, landmark2.y, landmark2.z
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def process_image(file_path):
    input_image = cv2.imread(file_path)
    input_image = cv2.resize(input_image, (500, 500))
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            wrist = hand_landmarks.landmark[0]
            finger_tips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]

            distances = [
                (f"Wrist to Finger {i+1}", euclidean_distance(wrist, tip))
                for i, tip in enumerate(finger_tips)
            ]

            mp_drawing.draw_landmarks(input_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = input_image.shape
            for i, (name, distance) in enumerate(distances):
                start_point = (int(wrist.x * w), int(wrist.y * h))
                end_point = (int(finger_tips[i].x * w), int(finger_tips[i].y * h))
                cv2.line(input_image, start_point, end_point, (255, 0, 0), 2)
                mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                cv2.putText(input_image, f"{distance:.3f}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            return input_image, distances
        else:
            return input_image, None

# Define the directory
dataset_dir = "D:/KULIAH/SKRIPSI/vm_project/sample_data"

# Get all image files from the directory
image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(image_files) < 6:
    print(f"Not enough image files in the directory. Found {len(image_files)} files.")
    exit()

# Choose 6 random image files
random_files = random.sample(image_files, 6)

# Process each image
results = []
for file in random_files:
    file_path = os.path.join(dataset_dir, file)
    processed_image, distances = process_image(file_path)
    results.append((file, processed_image, distances))

# Create a 2x3 grid of images
grid = np.zeros((1000, 1500, 3), dtype=np.uint8)
for i, (file, image, distances) in enumerate(results):
    row = i // 3
    col = i % 3
    grid[row*500:(row+1)*500, col*500:(col+1)*500] = image

    # Add file name and distances to the image
    cv2.putText(grid, file, (col*500 + 10, row*500 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if distances:
        for j, (name, distance) in enumerate(distances):
            cv2.putText(grid, f"{name}: {distance:.3f}", (col*500 + 10, row*500 + 60 + j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Display the grid
cv2.imshow("Hand Landmarks and Euclidean Distances", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the grid
cv2.imwrite("hand_landmarks_grid.jpg", grid)
print("Grid image saved as 'hand_landmarks_grid.jpg'")