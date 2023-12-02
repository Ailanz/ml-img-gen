import random

from keras.callbacks import EarlyStopping

from src.evaluation.EvaluateDenoiser import EvaluateDenoiser
from src.features import img_process
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.model import noise_predictor
import numpy as np


def add_noise(images, noise_factor=random.random()):
    # noisy_images = images + noise_factor * np.random.normal(loc=0., scale=1., size=images.shape)
    rand_scale = np.random.rand(images.shape[0])
    noisy_images = images + np.random.rand(images.shape[0], 128, 128, 3) * np.random.normal(loc=0., scale=1., size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    noise = noisy_images - images
    # rand_sample = random.randint(0, images.shape[0] - 1)
    # plot_img(noisy_images[rand_sample], images[rand_sample], noise[rand_sample])
    return noisy_images, noise

def plot_img(noisy_img, img, noise):
    f, axarr = plt.subplots(1, 4, figsize=(12, 12))
    axarr[0].imshow(noisy_img, interpolation='nearest')
    axarr[1].imshow(img, interpolation='nearest')
    axarr[2].imshow(noise, interpolation='nearest')
    axarr[3].imshow(noisy_img - noise, interpolation='nearest')
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[3].axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    denoiser = noise_predictor.NoisePredictor()
    model = denoiser.build_model()

    imgs = img_process.load_imgs_as_np(path='../../data/pokemon_aug_small/*.jpeg')
    imgs = imgs / 255.

    epoch_per_step = 5

    evaluate_denoiser = EvaluateDenoiser(evaluate_epochs=epoch_per_step, model=model)
    early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

    for i in range(10000):
        noisy_images = imgs
        for step in range(1):
            # noise_factor = (step + 1) / 10  # Gradually increase noise factor
            # noise_factor = 1.0  # Gradually increase noise factor
            # print(f"Training step {i+1}/1000", f"noise factor: {noise_factor}")
            # noisy_images, noises = add_noise(noisy_images, noise_factor=noise_factor)
            noisy_images, noises = add_noise(imgs)

            # y = model.predict(np.expand_dims(noisy_images[0], axis=0))
            # plot_img(noisy_images[0], imgs[0], noisy_images[0] - y[0])
            # noise_factor_batch = np.repeat(noise_factor, noisy_images.shape[0], axis=0)
            # model.fit([noisy_images, noise_factor_batch], noises, epochs=epoch_per_step, batch_size=32, verbose=1, callbacks=[early_stopping, evaluate_denoiser])
            rand_sample = random.randint(0, imgs.shape[0] - 1)
            plot_img(noisy_images[rand_sample], imgs[rand_sample], model.predict(np.expand_dims(noisy_images[rand_sample], axis=0))[0])
            model.fit(noisy_images, noises, epochs=epoch_per_step, batch_size=32, verbose=1, callbacks=[early_stopping, evaluate_denoiser])

    model.save('denoiser.h5')
    # time_steps = 5
    # for i in range(time_steps):
    #     print('Starting Time Step: ', i)
    #     noise_factor = i / (time_steps - 1)  # Gradually increase noise factor
    #     noisy_images, noises = add_noise(imgs, noise_factor=noise_factor)
    #     print(f"Training step {i+1}/{time_steps}, noise factor: {noise_factor}")
    #     model.fit(noisy_images, noises, epochs=10, batch_size=32, verbose=1, callbacks=[early_stopping, evaluate_denoiser])

    print("Done!")
