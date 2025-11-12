"""
    Followed this tutorial : https://www.datacamp.com/tutorial/face-detection-python-opencv
"""

import os
import cv2
import numpy as np
import kagglehub
import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split

# image_size = 64
# images = []

# path = kagglehub.dataset_download("jessicali9530/lfw-dataset")

# for root, dirs, files in os.walk(path):
#     for file in files:
#         if file.endswith(".jpg"):
#             img_path = os.path.join(root, file)
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (image_size, image_size))
#             img = img.astype("float32") / 255.0
#             images.append(img)


# Test image
im_path = "src/data/guy.jpg"
im = cv2.imread(im_path)
print(im.shape)

# Convert to grey
gray_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
print(gray_im.shape)

# Load face detection classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Call classifier on image to detect faces of any size
face = face_classifier.detectMultiScale(
    gray_im, scaleFactor=1.1 , minNeighbors=5 , minSize=(40,40) 
)

# TODO : Crop the image to the bounding box, and then scale the image to the desired resolution

# Draw a rectangle on the face
for (x, y, w, h) in face:
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 4)

im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
print(im_rgb.shape)
cv2.imwrite("src/data/newguy.jpg",im_rgb)

print("Done")
