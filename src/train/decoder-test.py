import random
from src.features import img_process
from src.model import clip_img_decoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np

model = keras.models.load_model('decoder.keras')

imgs = img_process.load_imgs(as_np=False)
img_features, text_features = img_process.img_embedding(imgs_arr=imgs)

imgs = img_process.load_imgs(as_np=True)
imgs = imgs / 255.

X_train, X_test, y_train, y_test = train_test_split(img_features, imgs, test_size=0.25, random_state=42)
rand_img = np.random.randint(0, 256, size=(256, 256, 3))
rand_img = rand_img / 255.
rand_img.reshape(1, 256, 256, 3)
y_pred = model.predict(X_test)
# y_pred = y_pred * 255.

# out = model.predict(X_train[0])
# plt.imshow(y_pred[0], interpolation='nearest')
# plt.show()

samples = 5
f, axarr = plt.subplots(samples, samples, figsize=(12, 12))

for i in range(samples):
    for j in range(samples):
        axarr[i, j].imshow(y_pred[random.randint(1, y_pred.shape[0]-1)], interpolation='nearest')
        axarr[i, j].axis('off')

plt.show()

print("Done!")