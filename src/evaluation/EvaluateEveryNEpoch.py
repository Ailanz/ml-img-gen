import random
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class EvaluateEveryNEpochs(Callback):
    def __init__(self, evaluate_epochs=10, original_img=None, img_features=None, model=None):
        super(EvaluateEveryNEpochs, self).__init__()
        self.evaluate_epochs = evaluate_epochs
        self.model = model
        self.img_features = img_features
        self.original_img = original_img

    def on_epoch_end(self, epoch, logs=None):
        print(f'\nEpoch {epoch + 1}')
        if (epoch + 1) % self.evaluate_epochs == 0:
            try:
                y_pred = self.model.predict(self.img_features)

                samples = 5
                f, axarr = plt.subplots(samples, samples, figsize=(12, 12))

                for i in range(samples):
                    for j in range(samples):
                        axarr[i, j].imshow(y_pred[random.randint(1, y_pred.shape[0]-1)], interpolation='nearest')
                        axarr[i, j].axis('off')

                plt.show()
            except Exception as e:
                print(e)
                print("Error in EvaluateEveryNEpochs")
