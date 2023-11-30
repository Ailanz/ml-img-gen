from keras.layers import Reshape, Attention, Conv1D, Conv2DTranspose
import tensorflow as tf
from numpy import mean

from src.features import img_process
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Attention
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
        self.vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
        self.vgg16.trainable = False
        # self.vgg16.summary()

    def predict(self, img_features):
        return self.model.predict(img_features)

    def train(self, img_features, imgs):
        self.model.fit(img_features, imgs, epochs=500, batch_size=28, verbose=1)

    def loss(self, y_true, y_pred):
        # true_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        y_true_resized = tf.image.resize(y_true, (8, 8))
        y_pred_resized = tf.image.resize(y_pred, (8, 8))
        true_loss = tf.keras.losses.mean_squared_error(y_true_resized, y_pred_resized)

        vgg_loss = tf.keras.losses.mean_squared_error(self.vgg16(y_true), self.vgg16(y_pred))
        # average the 2
        return true_loss * 0.5 + vgg_loss * 0.5

    def build_model(self, input_size=512, lr=0.2):
        input_layer = Input(shape=(input_size,))
        input = Reshape(target_shape=(8, 8, 8))(input_layer)

        # 16
        up1 = UpSampling2D()(input)
        up1 = concatenate([Attention()([up1, up1]), up1], axis=-1)
        # 32
        up2 = UpSampling2D()(up1)
        up2 = concatenate([Attention()([up2, up2]), up2], axis=-1)
        # 64
        up3 = UpSampling2D()(up2)
        up3 = concatenate([Attention()([up3, up3]), up3], axis=-1)
        # 128
        up4 = UpSampling2D()(up3)
        up4 = concatenate([Attention()([up4, up4]), up4], axis=-1)
        # 256
        up5 = UpSampling2D()(up4)
        up5 = concatenate([Attention()([up5, up5]), up5], axis=-1)

        x = Conv2D(512, 3, activation='relu', padding='same')(input)
        x = BatchNormalization()(x)
        # x = Conv2D(2028, 3, activation='relu', padding='same')(x)
        # x = BatchNormalization()(x)
        x = Conv2DTranspose(12, 3, strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # 16
        x = concatenate([up1, x], axis=-1)
        # x = Conv2D(1024, 3, activation='relu', padding='same')(x)
        # x = BatchNormalization()(x)
        x = Conv2D(512, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(12, 3, strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # 32`
        x = concatenate([up2, x], axis=-1)
        x = Conv2D(256, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(12, 3, strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # 64
        x = concatenate([up3, x], axis=-1)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(12, 3, strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # 128
        x = concatenate([up4, x], axis=-1)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(12, 3, strides=(2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # 256
        x = concatenate([up5, x], axis=-1)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)

        x = BatchNormalization()(x)
        x = Conv2D(3, 3, activation='sigmoid', padding='same')(x)

        model = Model(inputs=input_layer, outputs=x)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        self.model = model
        return model


if __name__ == '__main__':
    decoder = ClipImgDecoder()
    decoder.build_model(512)
    # decoder.build_unet(input_shape=(256, 256, 3))
