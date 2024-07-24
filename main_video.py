import threading
import cv2
from simple_facerec import SimpleFacerec

# Initialize face recognition
sfr = SimpleFacerec()
sfr.load_encoding_images(r'C:\Users\Noah Lee\OneDrive\Documents\GitHub\face_detection\data')

URL = "http://192.168.1.121:81/stream"

# Open video capture from the URL
cap = cv2.VideoCapture(URL)


counter = 0
face_locations = []
face_names = []
lock = threading.Lock()

# Function to check face match
def check_face(frame):
    global face_locations, face_names
    try:
        face_locations, face_names = sfr.detect_known_faces(frame)
    except Exception as e:
        print(f"Error in face detection: {e}")

# Main loop
while True:
    ret, frame = cap.read()
    if ret:
        if counter % 10 == 0:  # Run face detection every 30 iterations
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except Exception as e:
                print(f"Error in starting thread: {e}")

        counter += 1

        # Draw face locations and names
        with lock:
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
