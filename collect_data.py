#%%
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import cv2
import re
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
#%%
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
#%%
# Get the current working directory
desired_dir = r'C:\Users\Noah Lee\OneDrive\Documents\GitHub\face_detection'
os.chdir(desired_dir)

# Verify the working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")
#%%
# Setup paths relative to the current directory
POS_PATH = os.path.join(current_dir, 'data', 'positive')
NEG_PATH = os.path.join(current_dir, 'data', 'negative')
ANC_PATH = os.path.join(current_dir, 'data', 'anchor')
#%%
# Make the directories if they do not exist
if not os.path.exists(POS_PATH):
    os.makedirs(POS_PATH)

if not os.path.exists(NEG_PATH):
    os.makedirs(NEG_PATH)

if not os.path.exists(ANC_PATH):
    os.makedirs(ANC_PATH)
# #%%
# # http://vis-www.cs.umass.edu/lfw/
# # Uncompress Tar GZ Labelled Faces in the Wild Dataset
# !tar -xf lfw.tgz
# #%%
# # Move LFW Images to the following repository data/negative
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw', directory)):
#         EX_PATH = os.path.join('lfw', directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)
#%%
import uuid

#%%
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#%%
cap = cv2.VideoCapture(0)

center_x, center_y = 125, 125

while cap.isOpened(): 
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                center_x = x + w // 2
                center_y = y + h // 2
                # print(center_x)
                # print(center_y)
    cropped_frame = frame[center_y - 125: center_y + 125, center_x - 125: center_x + 125, :]
        
    #Collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        #Create unique name
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(imgname, cropped_frame)
        pass
        
    #Collect Positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        #Create unique name
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(imgname, cropped_frame)
        pass
        
    
    cv2.imshow('Image', frame)
    cv2.imshow('Cropped Image', cropped_frame)
    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()
# %%
plt.imshow(frame[120:120+250,200:250+250,:])
plt.imshow(frame)
# %%
