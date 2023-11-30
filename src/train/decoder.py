import random

from keras.callbacks import EarlyStopping

from src.evaluation.EvaluateEveryNEpoch import EvaluateEveryNEpochs
from src.features import img_process
from src.model import clip_img_decoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.model import clip_img_decoder

decoder = clip_img_decoder.ClipImgDecoder()
model = decoder.build_model()


imgs = img_process.load_imgs()
img_features, text_features = img_process.img_embedding(imgs_arr=imgs)

# model = decoder.build_unet(input_shape=(256, 256, 3))
imgs = img_process.load_imgs_as_np()
imgs = imgs / 255.

# decoder
X_train, X_test, y_train, y_test = train_test_split(img_features, imgs, test_size=0.2, random_state=42)

# unet
# X_train, X_test, y_train, y_test = train_test_split(imgs, imgs, test_size=0.33, random_state=42)

early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
evaluate_callback = EvaluateEveryNEpochs(evaluate_epochs=10, original_img=y_test, img_features=X_test, model=model)


model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=1, callbacks=[early_stopping, evaluate_callback])
model.save('decoder.keras')

print("Done!")
