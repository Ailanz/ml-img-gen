import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from PIL import Image

def add_noise(images, noise_factor=0.1):
    images_noisy = images + noise_factor * np.random.normal(loc=0., scale=1., size=images.shape)
    images_noisy = np.clip(images_noisy, 0., 1.)
    return images_noisy

def build_autoencoder():
    autoencoder = Sequential([
        Conv2D(64, 3, activation='relu', padding='same', input_shape=(128, 128, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu', padding='same'),
        UpSampling2D(),
        Conv2D(64, 3, activation='relu', padding='same'),
        UpSampling2D(),
        Conv2D(3, 3, activation='sigmoid', padding='same')
    ])
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
    return autoencoder


# Let's use the same autoencoder architecture
def build_autoencoder():
    autoencoder = Sequential([
        Conv2D(64, 3, activation='relu', padding='same', input_shape=(128, 128, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu', padding='same'),
        UpSampling2D(),
        Conv2D(64, 3, activation='relu', padding='same'),
        UpSampling2D(),
        Conv2D(3, 3, activation='sigmoid', padding='same')
    ])
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
    return autoencoder


# Load and preprocess images just like before
original_images = load_and_preprocess_images(my_image_paths)

# Initialize autoencoder
autoencoder = build_autoencoder()

# Train multiple timesteps
timesteps = 5
for i in range(timesteps):
    noise_factor = i / (timesteps - 1)  # Gradually increase noise factor
    noisy_images = add_noise(original_images, noise_factor=noise_factor)

    print(f"Training step {i+1}/{timesteps}, noise factor: {noise_factor}")

    autoencoder.fit(noisy_images, original_images, epochs=5, batch_size=32)

    if i != timesteps - 1:  # To avoid unnecessary model building
        # After each timestep, we rebuild the model to be without the last layers
        autoencoder = Model(autoencoder.input, autoencoder.layers[-3].output)
        autoencoder = Sequential(autoencoder.layers + [
            UpSampling2D(),
            Conv2D(64, 3, activation='relu', padding='same'),
            UpSampling2D(),
            Conv2D(3, 3, activation='sigmoid', padding='same')
        ])
        autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
