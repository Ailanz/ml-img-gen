from keras.layers import Reshape, Attention, Conv1D, Conv2DTranspose
import tensorflow as tf
from numpy import mean

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


class NoisePredictor:
    def __init__(self, raw_imgs=None):
        self.model = None
        self.raw_imgs = raw_imgs
        # self.vgg16.summary()

    def build_model(self, input_shape=(128, 128, 3), lr=0.2):
        # 16 = default
        muliple = 16
        inputs = Input(input_shape)
        inputs = Conv2D(3, 3, activation='sigmoid', padding='same')(inputs)

        down1 = MaxPooling2D(pool_size=(2, 2))(inputs)

        # pre = concatenate([Flatten()(inputs), inputs2])
        # pre = Dense(16, activation='relu')(pre)
        # pre = Dense(128 * 128 * 3, activation='relu')(pre)
        # pre = Reshape((128, 128, 3))(pre)
        # Contracting path
        conv1 = Conv2D(2 * muliple, 3, activation='relu', padding='same')(inputs)
        # conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(2 * muliple, 3, activation='relu', padding='same')(conv1)
        # conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = concatenate([pool1], axis=-1)
        conv2 = Conv2D(4 * muliple, 3, activation='relu', padding='same')(conv2)
        # conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(4 * muliple, 3, activation='relu', padding='same')(conv2)
        # conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bottleneck
        conv3 = concatenate([pool2], axis=-1)
        conv3 = Conv2D(8 * muliple, 3, activation='relu', padding='same')(conv3)
        # conv3 = BatchNormalization()(conv3)

        # Expanding path
        # up4 = Conv2DTranspose(3, 3, strides=(2, 2), activation='relu', padding='same')(conv3)
        up4 = UpSampling2D()(conv3)
        up4 = concatenate([up4, conv2, down1], axis=-1)
        conv4 = Conv2D(4 * muliple, 3, activation='relu', padding='same')(up4)
        # conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(4 * muliple, 3, activation='relu', padding='same')(conv4)
        # conv4 = BatchNormalization()(conv4)

        # up5 = Conv2DTranspose(3, 3, strides=(2, 2), activation='relu', padding='same')(conv4)
        up5 = UpSampling2D()(conv4)
        up5 = concatenate([up5, conv1, inputs], axis=-1)
        conv5 = Conv2D(2 * muliple, 3, activation='relu', padding='same')(up5)
        # conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(2 * muliple, 3, activation='relu', padding='same')(conv5)
        conv5 = Conv2D(1 * muliple, 3, activation='relu', padding='same')(conv5)
        # conv5 = BatchNormalization()(conv5)

        # Output layer
        output = Conv2D(3, (1, 1), activation='tanh')(conv5)
        model = Model(inputs=[inputs], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        # model.compile(optimizer='adam', loss='binary_crossentropy')
        model.summary()
        return model


if __name__ == '__main__':
    decoder = NoisePredictor()
    decoder.build_model()

    # decoder.build_unet(input_shape=(256, 256, 3))
