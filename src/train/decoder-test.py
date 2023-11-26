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

X_train, X_test, y_train, y_test = train_test_split(img_features, imgs, test_size=0.33, random_state=42)
rand_img = np.random.randint(0, 256, size=(256, 256, 3))
rand_img = rand_img / 255.
rand_img.reshape(1, 256, 256, 3)
y_pred = model.predict(X_test)
# y_pred = y_pred * 255.

# out = model.predict(X_train[0])
# plt.imshow(y_pred[0], interpolation='nearest')
# plt.show()

samples = 5
f, axarr = plt.subplots(samples, 2)

for i in range(5):
    number = random.randint(0, y_pred.shape[0])
    # imshow image from y_pred

    axarr[i, 0].imshow(y_pred[number])
    axarr[i, 1].imshow(y_pred[number+1])

plt.show()

print("Done!")