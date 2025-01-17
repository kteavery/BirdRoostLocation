"""Train the shallow CNN model on a single radar product.

Use command line arguments to select which radar product to train the model on.
Optionally input the location of the save file where the default is
model/radar_product/
Use an integer to select a radar_product from the following list:
    0 : Reflectivity
    1 : Velocity
    2 : Correlation Coefficient
    3 : Differential Reflectivity

Example command:
python train.py \
--radar_product=0 \
--log_path=model/Reflectivity/ \
--eval_increment=5 \
--num_iterations=2500 \
--checkpoint_frequency=100 \
--learning_rate=.001 \
--model=0 \
--dual_pol=True \
--high_memory_mode=True
"""
import argparse
import os, csv
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.BuildModels.ShallowCNN import unet as unet
from BirdRoostLocation.BuildModels.ShallowCNN import model as shallow_model
from keras.callbacks import History
from BirdRoostLocation import utils
from BirdRoostLocation.BuildModels import ml_utils
from BirdRoostLocation.ReadData import BatchGenerator
import tensorflow as tf
import datetime
import warnings

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D
from keras.models import model_from_json
from keras import Model
import keras

import matplotlib.pyplot as plt
import pandas
import numpy as np

warnings.simplefilter("ignore")


def train(
    log_path,
    radar_product,
    eval_increment=5,
    num_iterations=2500,
    checkpoint_frequency=100,
    lr=0.00001,
    model_name=utils.ML_Model.CNN,
    model_type="cnn",
    dual_pol=True,
    high_memory_mode=False,
    num_temporal_data=0,
    coord_conv=True,
    problem="detection",
    loadfile=0,
):
    """Train the shallow CNN model on a single radar product.

    Args:
        log_path: The location of the save directory. The model checkpoints,
            model weights, and the tensorboard events are all saved in this
            directory.
        radar_product: The radar product the model is training on. This should
            be a value of type utils.Radar_Products.
        eval_increment: How frequently the model prints checks validation result
        num_iterations: The number of training iterations the model will run.
        checkpoint_frequency: How many training iterations should the model
            perform before saving out a checkpoint of the model training.
        lr: The learning rate of the model, this value must be between 0 and 1.
            e.g. .1, .05, .001
        model_name: Select the model to train. Must be of type utils.ML_Model
        dual_pol: True if data training on dual polarization radar data, false
            when training on legacy data.
        high_memory_mode: True if training in high memory mode. High memory
            mode reduces the amount of IO operations by keeping all the data in
            memory during trainig. Not recommended for computes with fewer than
            8 GB of memory.
    """
    save_file = ml_utils.KERAS_SAVE_FILE.format(radar_product.fullname, "{}")

    checkpoint_path = log_path + ml_utils.CHECKPOINT_DIR
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path))

    with open(checkpoint_path + "train_log.csv", "w", newline="") as csvfile:
        train_writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        train_writer.writerow(["loss", "accuracy"])

    with open(checkpoint_path + "val_log.csv", "w", newline="") as csvfile:
        val_writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        val_writer.writerow(["loss", "accuracy"])

    print("MODEL NAME")
    print(model_name)
    print(settings.ML_SPLITS_DATA)

    if model_name == utils.ML_Model.CNN:
        print("utils.ML_Model.CNN")
        print(settings.ML_SPLITS_DATA)

        batch_generator = BatchGenerator.Single_Product_Batch_Generator(
            ml_label_csv=settings.LABEL_CSV,
            ml_split_csv=settings.ML_SPLITS_DATA,
            high_memory_mode=high_memory_mode,
        )
        if model_type == "unet":
            model = unet.build_model(
                inputDimensions=(240, 240, 3),
                lr=lr,
                coord_conv=coord_conv,
                problem=problem,
            )
        else:  # shallow CNN
            model = shallow_model.build_model(
                inputDimensions=(240, 240, 3),
                lr=lr,
                coord_conv=coord_conv,
                problem=problem,
            )

    elif model_name == utils.ML_Model.CNN_All:
        tf.compat.v1.disable_eager_execution()
        config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1, allow_soft_placement=True
        )
        tf_session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(tf_session)

        loaded_models = ml_utils.load_all_models(
            dual_pol, loadfile, problem="localization"
        )

        print(settings.ML_SPLITS_DATA)
        batch_generator = BatchGenerator.Multiple_Product_Batch_Generator(
            ml_label_csv=settings.LABEL_CSV,
            ml_split_csv=settings.ML_SPLITS_DATA,
            high_memory_mode=high_memory_mode,
        )

        if model_type == "unet":
            print("creating model")

            inputs = Input((4, 240, 240))
            conv1 = Conv2D(
                1056,
                3,
                data_format="channels_first",
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(inputs)
            conv2 = Conv2D(1, 1, data_format="channels_first", activation="sigmoid")(
                conv1
            )
            model = Model(inputs, conv2)
            model.compile(
                optimizer=keras.optimizers.adam(lr),
                loss=unet.dice_coef_loss,
                metrics=[unet.dice_coef],
            )

            print(model.summary())
        else:
            model = Sequential()
            model.add(Dense(256, input_shape=(4, 2), activation="relu"))
            model.add(Dense(2, activation="softmax"))
            model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss=keras.losses.categorical_crossentropy,
                metrics=["accuracy"],
            )

    print("checkpoint path: ")
    print(checkpoint_path)

    # Set up callback
    train_history = ml_utils.LossHistory()
    train_history.on_train_begin()
    val_history = ml_utils.LossHistory()
    val_history.on_train_begin()

    if problem == "detection":
        train_names = ["train_loss", "train_accuracy"]
        val_names = ["val_loss", "val_accuracy"]

        progress_string = "{} Epoch: {} Loss: {} Accuracy {}"

    else:  # location "mae", "mape", "cosine"
        train_names = ["train_mse", "train_mae", "train_mape", "train_cosine"]
        val_names = ["val_mse", "val_mae", "val_mape", "val_cosine"]

        progress_string = "{} Epoch: {} Loss: {} MAE: {} MAPE: {} Cosine: {}"

    for batch_no in range(num_iterations):
        x = None
        y = None
        while type(x) == type(None) and type(y) == type(None):
            # print(model_name)
            if model_name == utils.ML_Model.CNN:
                x, y, _ = batch_generator.get_batch(
                    ml_set=utils.ML_Set.training,
                    dualPol=dual_pol,
                    radar_product=radar_product,
                    num_temporal_data=num_temporal_data,
                    model_type=model_type,
                    problem=problem,
                )
                print(x.shape)
                if type(x) != type(None) and type(y) != type(None):
                    print("X and Y shapes")
                    print(x.shape)
                    print(y.shape)
                if problem == "localization" and type(y) != type(None):
                    y = np.reshape(y, (x.shape[0], x.shape[1], x.shape[2], 1))

            if model_name == utils.ML_Model.CNN_All:
                img_list, y, x, file_list = batch_generator.get_batch(
                    ml_set=utils.ML_Set.training,
                    dualPol=dual_pol,
                    loaded_models=loaded_models,
                    num_temporal_data=num_temporal_data,
                    model_type=model_type,
                    problem=problem,
                )
                if type(x) != type(None) and type(y) != type(None):
                    print(x.shape)
                    print(y.shape)
                    if problem == "detection":
                        x = np.reshape(x, (x.shape[1], x.shape[0], x.shape[2]))
                        y = np.reshape(y, (y.shape[1], y.shape[0], y.shape[2]))
                    else:
                        x = np.reshape(
                            x, (x.shape[1], x.shape[0], x.shape[2], x.shape[3])
                        )
                        y = y[0]
                        # y = np.reshape(
                        #    y, (y.shape[1], y.shape[0], y.shape[2], y.shape[3])
                        # )
                    print("train.py - x and y shapes")
                    print(x.shape)
        train_logs = model.train_on_batch(np.array(x), np.array(y))
        print(train_logs)

        train_history.on_batch_end(batch=(x, y), logs=train_logs)

        if problem == "detection":
            with open(checkpoint_path + "train_log.csv", "a") as csvfile:
                train_writer = csv.writer(
                    csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                train_writer.writerow([train_logs[0], train_logs[1]])

            print(
                progress_string.format(
                    utils.ML_Set.training.fullname,
                    batch_no,
                    train_logs[0],
                    train_logs[1],
                )
            )
        else:
            print(type(train_logs))

            if len(train_logs) == 4:
                with open(checkpoint_path + "train_log.csv", "a") as csvfile:
                    train_writer = csv.writer(
                        csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
                    )
                    train_writer.writerow(
                        [train_logs[0], train_logs[1], train_logs[2], train_logs[3]]
                    )

                print(
                    progress_string.format(
                        utils.ML_Set.training.fullname,
                        batch_no,
                        train_logs[0],
                        train_logs[1],
                        train_logs[2],
                        train_logs[3],
                    )
                )
            else:
                with open(checkpoint_path + "train_log.csv", "a") as csvfile:
                    train_writer = csv.writer(
                        csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
                    )
                    train_writer.writerow([train_logs[0], train_logs[1]])

                print(
                    progress_string.format(
                        utils.ML_Set.training.fullname,
                        batch_no,
                        train_logs[0],
                        train_logs[1],
                        None,
                        None,
                    )
                )

        # only print validation every once in a while
        if batch_no % eval_increment == 0:
            try:
                if model_name == utils.ML_Model.CNN_All:
                    print("before batch gen")
                    _, y_, x_, _ = batch_generator.get_batch(
                        ml_set=utils.ML_Set.validation,
                        dualPol=dual_pol,
                        loaded_models=loaded_models,
                        num_temporal_data=num_temporal_data,
                        model_type=model_type,
                        problem=problem,
                    )
                    print("batch gen created")
                else:
                    x_, y_, _ = batch_generator.get_batch(
                        ml_set=utils.ML_Set.validation,
                        dualPol=dual_pol,
                        radar_product=radar_product,
                        num_temporal_data=num_temporal_data,
                        model_type=model_type,
                        problem=problem,
                    )

                if problem == "localization":
                    if model_name == utils.ML_Model.CNN_All:
                        print(x_.shape)
                        print(y_.shape)
                        x_ = np.reshape(
                            x_, (x_.shape[1], x_.shape[0], x_.shape[2], x_.shape[3])
                        )
                        y_ = y_[0]
                        # y_ = np.reshape(
                        #    y_, (y_.shape[1], y_.shape[0], y_.shape[2], y_.shape[3])
                        # )
                        print("train.py - x and y shapes")
                        print(x_.shape)
                        print(y_.shape)
                    else:
                        y_ = np.reshape(y_, (x_.shape[0], x_.shape[1], x_.shape[2], 1))
                else:  # detection
                    if model_name == utils.ML_Model.CNN_All:
                        print(x_.shape)
                        print(y_.shape)
                        x_ = np.reshape(x_, (x_.shape[1], x_.shape[0], x_.shape[2]))
                        y_ = np.reshape(y_, (y_.shape[1], y_.shape[0], y_.shape[2]))
                        print("train.py - x and y shapes")
                        print(x_.shape)
                        print(y_.shape)
                    else:
                        # print("y_.shape")
                        y_ = np.reshape(y_, (x_.shape[0], 2))
                        # print(y_.shape)

                print("BEFORE model.test_on_batch")
                val_logs = model.test_on_batch(x_, y_)
                print("AFTER model.test_on_batch")

                print("problem == detection")
                print(problem)
                if problem == "detection":
                    with open(checkpoint_path + "val_log.csv", "a") as csvfile:
                        val_writer = csv.writer(
                            csvfile,
                            delimiter=" ",
                            quotechar="|",
                            quoting=csv.QUOTE_MINIMAL,
                        )
                        val_writer.writerow([val_logs[0], val_logs[1]])

                    print(
                        progress_string.format(
                            utils.ML_Set.validation.fullname,
                            batch_no,
                            val_logs[0],
                            val_logs[1],
                        )
                    )
                else:  # localization
                    val_history.on_batch_end(batch=(x, y), logs=val_logs)

                    print(len(val_logs))
                    if len(val_logs) == 4:
                        with open(checkpoint_path + "val_log.csv", "a") as csvfile:
                            val_writer = csv.writer(
                                csvfile,
                                delimiter=" ",
                                quotechar="|",
                                quoting=csv.QUOTE_MINIMAL,
                            )
                            val_writer.writerow(
                                [val_logs[0], val_logs[1], val_logs[2], val_logs[3]]
                            )

                        print(
                            progress_string.format(
                                utils.ML_Set.validation.fullname,
                                batch_no,
                                val_logs[0],
                                val_logs[1],
                                val_logs[2],
                                val_logs[3],
                            )
                        )
                    else:
                        with open(checkpoint_path + "val_log.csv", "a") as csvfile:
                            val_writer = csv.writer(
                                csvfile,
                                delimiter=" ",
                                quotechar="|",
                                quoting=csv.QUOTE_MINIMAL,
                            )
                            val_writer.writerow([val_logs[0], val_logs[1]])

                        print(
                            progress_string.format(
                                utils.ML_Set.validation.fullname,
                                batch_no,
                                val_logs[0],
                                val_logs[1],
                                None,
                                None,
                            )
                        )
                x_, y_, x, y = [None] * 4
                print("completed validation")

            except Exception as e:
                print(e)

        if batch_no % checkpoint_frequency == 0 or batch_no == num_iterations - 1:
            # currentDT = datetime.datetime.now()
            model_json = model.to_json()
            with open(
                checkpoint_path + "/" + save_file.format("") + ".json", "w"
            ) as json_file:
                json_file.write(model_json)

            model.save_weights(os.path.join(checkpoint_path, save_file.format("")))
            if unlabeled == "":
                try:
                    ml_utils.create_plots(
                        train=train_history,
                        val=val_history,
                        save_path=os.path.join(
                            checkpoint_path, "mse_plot_" + str(batch_no) + ".png"
                        ),
                    )
                except Exception as e:
                    print(e)

    print("SAVE FILE")
    print(save_file)
    # model.save_weights(save_file)


def main(results):
    os.chdir(settings.WORKING_DIRECTORY)
    radar_product = utils.Radar_Products(results.radar_product)
    model = utils.ML_Model(results.model)
    print("MODEL")
    print(model)
    if results.log_path is None:
        if results.model == 1:
            log_path = ml_utils.LOG_PATH.format(model.fullname, str(results.dual_pol))
        elif results.model == 0:
            log_path = ml_utils.LOG_PATH.format(model.fullname, radar_product.fullname)
        else:
            log_path = ml_utils.LOG_PATH_TIME.format(
                model.fullname,
                results.num_temporal_data * 2 + 1,
                radar_product.fullname,
            )
    else:
        log_path = results.log_path

    print("Log path: ")
    print(log_path)
    train(
        log_path=log_path,
        radar_product=radar_product,
        eval_increment=results.eval_increment,
        num_iterations=results.num_iterations,
        checkpoint_frequency=results.checkpoint_frequency,
        lr=results.learning_rate,
        model_name=model,
        model_type=results.model_type,
        dual_pol=results.dual_pol,
        high_memory_mode=results.high_memory_mode,
        num_temporal_data=results.num_temporal_data,
        coord_conv=results.coord_conv,
        problem=results.problem,
        loadfile=results.loadfile,
    )


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--radar_product",
        type=int,
        default=1,
        help="""
            Use an integer to select a radar_product from the following list:
                0 : Reflectivity
                1 : Velocity
                2 : Correlation Coefficient
                3 : Differential Reflectivity
            """,
    )
    parser.add_argument(
        "-l",
        "--log_path",
        type=str,
        default=None,
        help="""
            Optionally input the location of the save file where the default is
            model/radar_product
            """,
    )
    parser.add_argument(
        "-e",
        "--eval_increment",
        type=int,
        default=5,
        help="""How frequently the model prints out the validation results.""",
    )
    parser.add_argument(
        "-n",
        "--num_iterations",
        type=int,
        default=2500,
        help="""The number of training iterations the model will run""",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_frequency",
        type=int,
        default=300,
        help="""
            How many training iterations should the model perform before saving 
            out a checkpoint of the model training.
            """,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0001,
        help="""
            The learning rate of the model, this value must be between 0 and 1
            .e.g. .1, .05, .001
            """,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=int,
        default=0,
        help="""
            Use an integer to select a model from the following list:
                0 : Shallow CNN
                1 : Shallow CNN, all radar products
                2 : Shallow CNN, temporal model
            """,
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        default="shallow_model",
        help="""
            shallow_model
            unet
            """,
    )
    parser.add_argument(
        "-d",
        "--dual_pol",
        type=bool,
        default=True,
        help="""
            This field will only be used if model = 1 
            True if model is training on dual polarization radar data, false if 
            the model is training on legacy data.
            """,
    )
    parser.add_argument(
        "-hm",
        "--high_memory_mode",
        type=bool,
        default=False,
        help="""
            If true then all of the data will be read in at the beginning and 
            stored in memory. Otherwise only one batch of data will be in 
            memory at a time. high_memory_mode is good for machines with slow 
            IO and at least 8 GB of memory available.
            """,
    )
    parser.add_argument(
        "-td",
        "--num_temporal_data",
        type=int,
        default=1,
        help="""
                Only applied to temporal model. This indicates how many time
                frames in either direction used for training. 0 will give array
                size of 1, 1 -> 3, 2 -> 5, and 3 -> 7.
                """,
    )
    parser.add_argument(
        "-cc",
        "--coord_conv",
        type=bool,
        default=True,
        help="""
            Turn coord_conv layers on and off. See model.py.
            """,
    )
    parser.add_argument(
        "-p",
        "--problem",
        type=str,
        default="detection",
        help="""
            Type of problem to solve. Either 'detection' or 'localization'.
            """,
    )
    parser.add_argument(
        "-lf",
        "--loadfile",
        type=int,
        default=0,
        help="""
            which file to load for aggregates
            """,
    )
    results = parser.parse_args()
    main(results)
