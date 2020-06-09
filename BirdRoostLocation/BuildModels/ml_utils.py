import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import model_from_json
from BirdRoostLocation import utils
import BirdRoostLocation.LoadSettings as settings

KERAS_SAVE_FILE = "{}{}.h5"
LOG_PATH = "model/{}/{}/"
LOG_PATH_TIME = "model/{}/{}/{}/"
CHECKPOINT_DIR = "/checkpoint/"


def load_all_models(dual_pol, loadfile):
    loaded_models = []

    if dual_pol:
        radar_products = [
            utils.Radar_Products.cc,
            utils.Radar_Products.diff_reflectivity,
            utils.Radar_Products.reflectivity,
            utils.Radar_Products.velocity,
        ]
    else:
        radar_products = utils.Legacy_radar_products

    for radar_product in radar_products:
        if str(radar_product) == "Radar_Products.cc":
            product_str = "Rho_HV"
        elif str(radar_product) == "Radar_Products.diff_reflectivity":
            product_str = "Zdr"
        elif str(radar_product) == "Radar_Products.reflectivity":
            product_str = "Reflectivity"
        else:
            product_str = "Velocity"

        json_file = open(
            settings.WORKING_DIRECTORY
            + "model/"
            + str(product_str)
            + "/"
            + str(loadfile)
            + "/checkpoint/"
            + str(product_str)
            + ".json",
            "r",
        )
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        print(
            settings.WORKING_DIRECTORY
            + "model/"
            + str(product_str)
            + "/"
            + str(loadfile)
            + "/checkpoint/"
            + str(product_str)
            + ".h5"
        )
        model.load_weights(
            settings.WORKING_DIRECTORY
            + "model/"
            + str(product_str)
            + "/"
            + str(loadfile)
            + "/checkpoint/"
            + str(product_str)
            + ".h5"
        )
        loaded_models.append(model)

    return loaded_models


def create_plots(train, val, save_path):
    x_train = list(range(0, len(train.mse)))
    x_val = list(range(0, len(train.mse), 5))
    if len(x_val) > len(val.mse):
        x_val = x_val[-1]
    if len(x_val) < len(val.mse):
        val.mse = val.mse[-1]

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
        print(logs)
        print(len(logs))
        self.mse.append(logs[0])
        self.mae.append(logs[1])
        if len(logs) >= 3:
            self.mape.append(logs[2])
        if len(logs) >= 4:
            self.cosine.append(logs[3])
