
import os
import cv2
import numpy as np
import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


latent_dim = 128 
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


images = np.array(images)
print("Dataset final shape:", images.shape)


x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)

print("Training shape:", x_train.shape)
print("Test shape:", x_test.shape)

# Create encoder and decoder
encoder = keras.models.Sequential([
    keras.layers.Input((64, 64, 3)),
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'), # I think that if we change the image size we have to change this value
    keras.layers.Dense(latent_dim * 2)
], name="encoder")
encoder.summary()

decoder = keras.models.Sequential([
    keras.layers.Input((latent_dim,)),
    keras.layers.Dense(8 * 8 * 128, activation='relu'),
    keras.layers.Reshape((8, 8, 128)),
    keras.layers.Conv2DTranspose(128, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2DTranspose(64, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same', strides=2),
    keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')
], name="decoder")
decoder.summary()

def sample_latent(z):
    z_mean, z_log_var = tf.split(z, num_or_size_splits=2, axis=1)
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    return z, z_mean, z_log_var

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        # Forward pass para predicción / validation
        z_all = self.encoder(inputs)
        z, _, _ = sample_latent(z_all)
        reconstruction = self.decoder(z)
        return reconstruction
    
vae = VAE(encoder, decoder)

def vae_loss(y_true, y_pred):
    # Reconstrucción
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred))
    
    # KL divergence
    z_all = vae.encoder(y_true)
    z_mean, z_log_var = tf.split(z_all, num_or_size_splits=2, axis=1)
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    )
    
    return reconstruction_loss + kl_loss

vae.compile(optimizer=keras.optimizers.Adam(), loss=vae_loss)

# Train the vae
epochs = 5
batch_size = 128
vae.fit(
    x_train, x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Compare original and reconstructed images
import matplotlib.pyplot as plt
import numpy as np

n = 5
z_all = vae.encoder(x_test[:n])
z, _, _ = sample_latent(z_all)
decoded_imgs = vae.decoder(z)

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