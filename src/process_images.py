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

image_size = 64
images = []

path = kagglehub.dataset_download("jessicali9530/lfw-dataset")

# Load face detection classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            # Convert to grey for face detection
            grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Call classifier on image to detect faces of any size
            face = face_classifier.detectMultiScale(
                grey_img, scaleFactor=1.1 , minNeighbors=5 , minSize=(40,40) 
            )

            # Crop the image to the bounding box, and then scale the image to the desired resolution
            for (x, y, w, h) in face : 
                x_max = x+w
                y_max = y+h
                im_cropped = img[y:y_max , x:x_max]


            im_cropped = cv2.cvtColor(im_cropped, cv2.COLOR_BGR2RGB)
            im_cropped = cv2.resize(im_cropped, (image_size, image_size))
            im_cropped = im_cropped.astype("float32") / 255.0
            images.append(im_cropped)
