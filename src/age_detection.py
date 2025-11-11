"""
This is a first test script that will be guided by this tutorial video : https://www.youtube.com/watch?v=ivFuOQcBiN8

The objective is to detect gender and approximate age on a given face. This may or may not be useful for the rest need to see. 
This is mostly just to familiarize ourselves with image based AI and commonly used libraries.

Dependencies : - OpenCV 

"""

import os
import cv2
import numpy as np
import kagglehub
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

path = kagglehub.dataset_download("jessicali9530/lfw-dataset")

image_size = 64
images = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            img = img.astype("float32") / 255.0
            images.append(img)

images = np.array(images)
print("Shape final del dataset:", images.shape)


x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)

print("Shape de entrenamiento:", x_train.shape)
print("Shape de prueba:", x_test.shape)

# Create encoder and decoder
encoder = keras.models.Sequential([
    keras.layers.Input((64, 64, 3)),
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', strides=2)
])

decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(128, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2DTranspose(64, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')  # salida RGB
])

autoencoder = keras.models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()


# Train the autoencoder
epochs = 10
batch_size = 128
autoencoder.fit(
    x_train, x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Compare original and reconstructed images
import matplotlib.pyplot as plt
import numpy as np

n = 1
decoded_imgs = autoencoder.predict(x_test[:n])

plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.axis("off")
plt.show()