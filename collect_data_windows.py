import os
import uuid
import cv2
import numpy as np
import tensorflow as tf

base_name = "Noah"

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Get the current working directory
desired_dir = r'C:\Users\Noah Lee\OneDrive\Documents\GitHub\face_detection'
os.chdir(desired_dir)

# Verify the working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Setup paths relative to the current directory
POS_PATH = os.path.join(current_dir, 'data', 'positive')
NEG_PATH = os.path.join(current_dir, 'data', 'negative')
ANC_PATH = os.path.join(current_dir, 'data', 'anchor')

# Make the directories if they do not exist
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)

total_images = 600
images_per_cycle = 10
number_of_cycles = total_images // (images_per_cycle * 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture image or captured an empty frame.")
        continue

    cv2.putText(frame, 'Ready? Press "Q" to start capturing', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(25)
    if key == ord('q'):
        break
    if key == ord('t'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

for cycle in range(number_of_cycles):
    for j in range(2):  # 0 for positive, 1 for anchor
        class_dir = POS_PATH if j == 0 else ANC_PATH
        existing_images = len(os.listdir(class_dir))  # Count existing images to avoid overwriting

        print(f'Collecting {images_per_cycle} images for class {j} at {class_dir}')

        counter = 0
        while counter < images_per_cycle:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to capture image or captured an empty frame.")
                continue

            # Get the frame dimensions
            h, w = frame.shape[:2]

            # Calculate the center of the frame
            center_x, center_y = w // 2, h // 2

            # Define the size of the cropped region
            crop_size = 250
            half_crop_size = crop_size // 2

            # Calculate the cropping coordinates
            x1 = max(center_x - half_crop_size, 0)
            y1 = max(center_y - half_crop_size, 0)
            x2 = min(center_x + half_crop_size, w)
            y2 = min(center_y + half_crop_size, h)

            # Crop the center region
            cropped_frame = frame[y1:y2, x1:x2]

            # Resize the cropped frame to ensure it is 250x250 pixels
            resized_frame = cv2.resize(cropped_frame, (250, 250))

            # Display the resized frame
            cv2.imshow('Resized Frame', resized_frame)

            # Save the resized frame
            img_filename = os.path.join(class_dir, '{}_{:04d}.jpg'.format(base_name, existing_images + counter))
            cv2.imwrite(img_filename, resized_frame)

            counter += 1

            # Break out of the loop and cancel code if 'r' is pressed
            if cv2.waitKey(10) & 0xff == ord('r'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()
