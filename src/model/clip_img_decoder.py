from keras.layers import Reshape, Attention, Conv1D, Conv2DTranspose

from src.features import img_process
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
# import sequential layer
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
import numpy as np


class ClipImgDecoder:
    def __init__(self, raw_imgs=None):
        self.model = None
        self.raw_imgs = raw_imgs

    def predict(self, img_features):
        return self.model.predict(img_features)

    def train(self, img_features, imgs):
        self.model.fit(img_features, imgs, epochs=500, batch_size=28, verbose=1)

    def build_model(self, input_size=512):
        input = Input(shape=(input_size,))
        x = Dense(1024, activation='relu')(input)
        x = Dropout(0.5)(x)
        x = Reshape(target_shape=(32, 32, 1))(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Dropout(0.5)(x)
        x = UpSampling2D((2, 2))(x)
        # x = concatenate([UpSampling2D(size=(2, 2))(x), input], axis=-1)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Dropout(0.5)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = Dropout(0.5)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, 3, activation='sigmoid', padding='same')(x)
        model = Model(inputs=input, outputs=x)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        self.model = model
        return model

    def build_model2(self, input_size=512, lr=0.2):

        input = Input(shape=(input_size,))
        # attention = Attention()([input, input])
        x = Dense(1024, activation='relu')(input)
        x = Dropout(lr)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(lr)(x)

        x = Reshape(target_shape=(32, 32, 1))(x)
        x = Conv2D(512, 3, activation='relu', padding='same')(x)
        x = Dropout(lr)(x)
        x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = Conv2D(256, 3, activation='relu', padding='same')(x)
        x = Dropout(lr)(x)
        x = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Dropout(lr)(x)
        x = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Dropout(lr)(x)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = Dropout(lr)(x)
        x = Conv2D(3, 3, activation='relu', padding='same')(x)
        x = Conv1D(3, 3, activation='relu', padding='same')(x)
        model = Model(inputs=input, outputs=x)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        self.model = model
        return model

    def build_unet(self, input_shape=(128, 128, 1)):
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
        output = Conv2D(3, (1, 1), activation='sigmoid')(conv5)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model


if __name__ == '__main__':
    decoder = ClipImgDecoder()
    decoder.build_model2(512)
    # decoder.build_unet(input_shape=(256, 256, 3))
