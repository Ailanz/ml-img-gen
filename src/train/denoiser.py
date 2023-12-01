import random

from keras.callbacks import EarlyStopping

from src.evaluation.EvaluateDenoiser import EvaluateDenoiser
from src.features import img_process
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.model import noise_predictor
import numpy as np


def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * np.random.normal(loc=0., scale=1., size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    noise = noisy_images - images
    # plot_img(noisy_images[0], images[0], noise[0])
    return noisy_images, noise

def plot_img(noisy_img, img, noise):
    f, axarr = plt.subplots(1, 3, figsize=(12, 12))
    axarr[0].imshow(noisy_img, interpolation='nearest')
    axarr[1].imshow(img, interpolation='nearest')
    axarr[2].imshow(noise, interpolation='nearest')
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    plt.show()



if __name__ == '__main__':
    denoiser = noise_predictor.NoisePredictor()
    model = denoiser.build_model()

    imgs = img_process.load_imgs_as_np(path='../../data/pokemon_aug_small/*.jpeg')
    imgs = imgs / 255.

    epoch_per_step = 5

    evaluate_denoiser = EvaluateDenoiser(evaluate_epochs=epoch_per_step, model=model)
    early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

    for i in range(1000):
        for step in range(10):
            noise_factor = (step + 1) / 10  # Gradually increase noise factor
            print(f"Training step {i+1}/1000", f"noise factor: {noise_factor}")
            noisy_images, noises = add_noise(imgs, noise_factor=noise_factor)

            # y = model.predict(np.expand_dims(noisy_images[0], axis=0))
            # plot_img(noisy_images[0], imgs[0], noisy_images[0] - y[0])
            noise_factor_batch = np.repeat(noise_factor, noisy_images.shape[0], axis=0)
            model.fit([noisy_images, noise_factor_batch], noises, epochs=epoch_per_step, batch_size=32, verbose=1, callbacks=[early_stopping, evaluate_denoiser])

    model.save('denoiser.h5')
    # time_steps = 5
    # for i in range(time_steps):
    #     print('Starting Time Step: ', i)
    #     noise_factor = i / (time_steps - 1)  # Gradually increase noise factor
    #     noisy_images, noises = add_noise(imgs, noise_factor=noise_factor)
    #     print(f"Training step {i+1}/{time_steps}, noise factor: {noise_factor}")
    #     model.fit(noisy_images, noises, epochs=10, batch_size=32, verbose=1, callbacks=[early_stopping, evaluate_denoiser])

    print("Done!")
