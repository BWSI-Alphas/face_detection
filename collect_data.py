import os
import cv2
import time

# Setup the data directory
DATA_PATH = r'C:\Users\Noah Lee\OneDrive\Documents\GitHub\face_detection\data'
os.makedirs(DATA_PATH, exist_ok=True)

URL = "http://192.168.1.121:81/stream"

# Open video capture from the URL
cap = cv2.VideoCapture(URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

# User's name for the file
user_name = "Jonah"  # Change this to the desired name

# Number of images to capture
num_images = 1
interval = 0.5  # Interval in seconds

image_counter = 0
capture_images = False
start_time = time.time()

while True:
    ret, frame = cap.read()
    if ret:
        # Get the frame dimensions
        h, w = frame.shape[:2]

        # Calculate the center of the frame
        center_x, center_y = w // 2, h // 2

        # Define the size of the cropped region
        crop_size = 400
        half_crop_size = crop_size // 2

        # Calculate the cropping coordinates
        x1 = max(center_x - half_crop_size, 0)
        y1 = max(center_y - half_crop_size, 0)
        x2 = min(center_x + half_crop_size, w)
        y2 = min(center_y + half_crop_size, h)

        # Crop the center region
        cropped_frame = frame[y1:y2, x1:x2]

        # Resize the cropped frame to ensure it is 400x400 pixels
        resized_frame = cv2.resize(cropped_frame, (400, 400))

        # Show the cropped frame
        cv2.imshow('Cropped Frame', resized_frame)

        # Capture images every 0.5 seconds
        if capture_images and time.time() - start_time >= interval:
            if image_counter < num_images:
                # Save the frame to the 'data' folder
                filename = os.path.join(DATA_PATH, f'{user_name}_{image_counter + 1}.jpg')
                cv2.imwrite(filename, resized_frame)
                print(f'Saved {filename}')
                
                image_counter += 1
                start_time = time.time()
            else:
                capture_images = False
                print("Finished capturing images.")

        # Show the continuous stream
        cv2.imshow('Live Stream', frame)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        capture_images = True
        image_counter = 0
        start_time = time.time()

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
