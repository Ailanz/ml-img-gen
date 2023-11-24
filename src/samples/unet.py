from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def create_unet(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)

    # Expanding path
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Output layer
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    return Model(inputs=inputs, outputs=output)

# model = create_unet(input_shape=(128, 128, 1))
# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.summary()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# We simply categorize pixels as being in the digit (label 1) or not (label 0)
train_labels = (train_images > 0.5).astype(np.int32)
test_labels = (test_images > 0.5).astype(np.int32)

# Create U-Net model
model = create_unet(input_shape=(28, 28, 1))

# Compile and train the model for 5 epochs
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(train_images, train_labels, epochs=5, batch_size=512)
model.save('unet.h5')
# Evaluate the trained U-Net model
test_loss = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)
