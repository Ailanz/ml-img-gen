# import mnist
import random

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# load mnist data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# load unet.h5
model = load_model('unet.h5')


number = random.randint(0, 10000)
# predict
pred = model.predict(test_images[number].reshape(1, 28, 28, 1))
print(pred.shape)

samples = 5
f, axarr = plt.subplots(samples, 2)

for i in range(5):
    number = random.randint(0, 10000)
    pred = model.predict(test_images[number].reshape(1, 28, 28, 1))
    axarr[i, 0].imshow(test_images[number], cmap='gray')
    axarr[i, 1].imshow(pred[0,:,:,0], cmap='gray')

plt.show()
