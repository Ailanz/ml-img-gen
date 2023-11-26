import random
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class EvaluateEveryNEpochs(Callback):
    def __init__(self, evaluate_epochs=10, img_features=None, model=None):
        super(EvaluateEveryNEpochs, self).__init__()
        self.evaluate_epochs = evaluate_epochs
        self.model = model
        self.img_features = img_features

    def on_epoch_end(self, epoch, logs=None):
        print(f'\nEpoch {epoch + 1}')
        if (epoch + 1) % self.evaluate_epochs == 0:
            y_pred = self.model.predict(self.img_features)

            samples = 5
            f, axarr = plt.subplots(samples, 2)

            for i in range(5):
                number = random.randint(0, y_pred.shape[0])
                axarr[i, 0].imshow(y_pred[number])
                axarr[i, 1].imshow(y_pred[number+1])

            plt.show()
