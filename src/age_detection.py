"""
This is a first test script that will be guided by this tutorial video : https://www.youtube.com/watch?v=ivFuOQcBiN8

The objective is to detect gender and approximate age on a given face. This may or may not be useful for the rest need to see. 
This is mostly just to familiarize ourselves with image based AI and commonly used libraries.

Dependencies : - OpenCV 

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# Load data TODO: only for testing purposes we are using the CIFAR10 dataset. Later it should be replaced by a proper face dataset (CelebA maybe).
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Shape de entrenamiento:", x_train.shape)
print("Shape de prueba:", x_test.shape)

# Create encoder and decoder
encoder = keras.models.Sequential([
    keras.layers.Input((32, 32, 3)),
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2,2), padding='same'),
    keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2,2), padding='same')
])

decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(16, (3,3), strides=2, activation='relu', padding='same'),
    keras.layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same'),
    keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')  # reconstrucción final
])

autoencoder = keras.models.Sequential([encoder, decoder])
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mse')

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