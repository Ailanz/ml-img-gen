import os
import random
import sys

import matplotlib.pyplot as plt
from keras.callbacks import Callback
import numpy as np


class EvaluateDenoiser(Callback):
    def __init__(self, evaluate_epochs=10, steps=10, model=None, shape=(128, 128, 3)):
        super(EvaluateDenoiser, self).__init__()
        self.evaluate_epochs = evaluate_epochs
        self.model = model
        self.shape = shape
        self.steps = steps

    def on_epoch_end(self, epoch, logs=None):
        print(f'\nEpoch {epoch + 1}')
        if (epoch + 1) % self.evaluate_epochs == 0:
            try:
                samples = 3

                noisy_img = np.random.normal(0.5, size=(samples, *self.shape))
                noisy_img = np.clip(noisy_img, 0., 1.)
                imgs = [noisy_img]
                for i in range(self.steps):
                    y_pred = self.model.predict(noisy_img)
                    imgs.append(y_pred)
                    noisy_img = y_pred

                f, axarr = plt.subplots(samples, self.steps, figsize=(12, 12))

                for i in range(samples):
                    for j in range(self.steps):
                        axarr[i, j].imshow(imgs[j][i], interpolation='nearest')
                        axarr[i, j].axis('off')

                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(e)
                print(e.__traceback__)
                print("Error in EvaluateEveryNEpochs")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
