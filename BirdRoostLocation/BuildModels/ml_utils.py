import tensorflow as tf
import matplotlib.pyplot as plt
import keras

KERAS_SAVE_FILE = "{}{}.h5"
LOG_PATH = "model/{}/{}/"
LOG_PATH_TIME = "model/{}/{}/{}/"
CHECKPOINT_DIR = "/checkpoint/"

def create_plots(train, val, save_path): 
    x_train = list(range(0,len(train.mse)))
    x_val = list(range(0,len(train.mse), 5))

    plt.plot(x_train, train.mse)
    plt.plot(x_val, val.mse)
    plt.title("model mse")
    plt.ylabel("mse")
    plt.xlabel("epoch")
    plt.legend(["training", "validation"], loc="upper left")
    plt.savefig(save_path)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.mse = []
        self.mae = []
        self.mape = []
        self.cosine = []

    def on_batch_end(self, batch, logs={}):
        self.mse.append(logs[0])
        self.mae.append(logs[1])
        self.mape.append(logs[2]]
        self.cosine.append(logs[3])

