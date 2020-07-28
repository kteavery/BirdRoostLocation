"""Evaluate the shallow CNN model trained on a single radar product.

Use command line arguments to select which radar product model to evaluate.
Optionally input the location of the save file where the default is
model/radar_product/
Use an integer to select a radar_product from the following list:
    0 : Reflectivity
    1 : Velocity
    2 : Correlation Coefficient
    3 : Differential Reflectivity

Example command:
python eval.py \
--radar_product=0 \
--log_path=model/Reflectivity/Reflectivity.h5

"""
import argparse
import os
import numpy as np
import csv
import ntpath
import cv2
import pandas as pd
from ast import literal_eval
import glob
from PIL import Image

import BirdRoostLocation.BuildModels.ShallowCNN.model as ml_model
import BirdRoostLocation.BuildModels.ShallowCNN.unet as unet
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation import utils
from BirdRoostLocation.BuildModels import ml_utils
from BirdRoostLocation.ReadData import BatchGenerator
from BirdRoostLocation.Analysis import SkillScores
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D
from keras import Model
import keras
from PIL import Image
import matplotlib


def field_predict(x, log_path, coord_conv, problem):
    if problem == "detection":
        model = ml_model.build_model(
            inputDimensions=(240, 240, 3), coord_conv=coord_conv, problem=problem
        )
        model.load_weights(log_path)
        predictions = model.predict(x)
    else:
        model = unet.build_model(
            inputDimensions=(240, 240, 3), coord_conv=coord_conv, problem=problem
        )
        print(log_path)
        model.load_weights(log_path)
        predictions = np.array([])
        for example in x:
            predictions = np.append(
                predictions, model.predict(np.reshape(example, (1, 240, 240, 3)))
            )
            print("predictions.shape")
            print(predictions.shape)
            predictions = np.reshape(predictions, (-1, 240, 240))
            print(predictions.shape)

    return predictions, model


def eval(
    log_path,
    radar_product,
    coord_conv,
    dual_pol,
    num_temporal_data,
    problem,
    model_name,
    loadfile=None,
    lr=0.00001,
    unlabeled="",
):
    """Evaluate the shallow CNN model trained on a single radar product.

        Args:
            log_path: The location of the save directory. This method will
                read the save located in this directory.
            radar_product: The radar product the model is evaluating. This
                should be a value of type utils.Radar_Products.
    """
    model_file = os.path.splitext(ntpath.basename(log_path))[0]
    batch_generator = BatchGenerator.Single_Product_Batch_Generator(
        ml_label_csv=settings.LABEL_CSV,
        ml_split_csv=settings.ML_SPLITS_DATA,
        validate_k_index=3,
        test_k_index=4,
        default_batch_size=200,
    )

    x = None
    y = None
    print("batch generator created")
    while type(x) == type(None) and type(y) == type(None):
        try:
            if model_name == utils.ML_Model.Shallow_CNN:
                print("unlabeled")
                print(unlabeled)
                if unlabeled == "":
                    x, y, filenames = batch_generator.get_batch(
                        utils.ML_Set.testing,
                        dualPol=dual_pol,
                        radar_product=radar_product,
                        num_temporal_data=num_temporal_data,
                        problem=problem,
                        is_eval=True,
                    )
                else:
                    y = None

                    filelist = glob.glob(unlabeled + "/*.png")
                    x = np.array(
                        [
                            np.array(Image.open(fname).convert("RGB"))
                            for fname in filelist
                        ]
                    )
                    # x = np.array([Image.fromarray(img).convert('RGB') for img in x])
                    x = x[:, 5:245, 5:245]
                    filenames = np.array(
                        [
                            os.path.splitext(os.path.basename(fname))[0]
                            for fname in filelist
                        ]
                    )

                print("x, y, filenames, predictions")
                print(x.shape)
                # print(y.shape)
                print(filenames.shape)
                predictions, model = field_predict(x, log_path, coord_conv, problem)
                print(predictions.shape)

            else:
                field_ys = np.array([])
                field_preds = np.array([])
                filenames = []

                for i, field in enumerate(
                    ["Zdr", "Rho_HV", "Velocity", "Reflectivity"]
                ):
                    if field == "Rho_HV":
                        radar_product = utils.Radar_Products.cc
                    elif field == "Zdr":
                        radar_product = utils.Radar_Products.diff_reflectivity
                    elif field == "Reflectivity":
                        radar_product = utils.Radar_Products.reflectivity
                    else:
                        radar_product = utils.Radar_Products.velocity

                    x, y, filenames = batch_generator.get_batch(
                        utils.ML_Set.testing,
                        dualPol=dual_pol,
                        radar_product=radar_product,
                        num_temporal_data=num_temporal_data,
                        problem=problem,
                        filenames=list(set(filenames)),
                        is_eval=True,
                    )

                    print("X, Y, Filenames: ")
                    print(x.shape)
                    print(y.shape)
                    print(filenames.shape)
                    print(model_name)

                    if problem == "detection":
                        preds, model = field_predict(
                            x,
                            settings.WORKING_DIRECTORY
                            + "model/"
                            + field
                            + "/"
                            + str(loadfile)
                            + "/checkpoint/"
                            + field
                            + ".h5",
                            coord_conv,
                            problem,
                        )
                        print(SkillScores.get_skill_scores(preds, y))
                    else:
                        preds, model = field_predict(
                            x,
                            settings.WORKING_DIRECTORY
                            + "model/"
                            + field
                            + "/unet/"
                            + str(loadfile)
                            + "/checkpoint/"
                            + field
                            + ".h5",
                            coord_conv,
                            problem,
                        )
                        print(SkillScores.get_skill_scores_localization(preds, y))

                    if field_preds.size == 0:
                        field_preds = preds
                        field_ys = y
                    else:
                        field_preds = np.concatenate((field_preds, preds), axis=0)
                        field_ys = np.concatenate((field_ys, y), axis=0)
                    print(field_preds.shape)
                    print(field_ys.shape)

                if problem == "detection":
                    model = Sequential()
                    model.add(Dense(256, input_shape=(4, 2), activation="relu"))
                    model.add(Dense(2, activation="softmax"))
                    model.compile(
                        loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.adam(lr),
                        metrics=["accuracy"],
                    )

                    model.load_weights(log_path)
                    print(field_preds.shape)
                    print(field_ys.shape)
                    predictions = model.predict(np.reshape(field_preds, (-1, 4, 2)))
                    # predictions = np.reshape(field_preds, (preds.shape[0], 4, 4))
                    if unlabeled == "":
                        y = np.reshape(field_ys, (-1, 4, 2))

                else:
                    inputs = Input((4, 240, 240))
                    conv1 = Conv2D(
                        1056,
                        3,
                        data_format="channels_first",
                        activation="relu",
                        padding="same",
                        kernel_initializer="he_normal",
                    )(inputs)
                    conv2 = Conv2D(1, 1, data_format="channels_first", activation="sigmoid")(conv1)
                    model = Model(inputs, conv2)
                    model.compile(
                        optimizer=keras.optimizers.adam(lr),
                        loss=unet.dice_coef_loss,
                        metrics=[unet.dice_coef],
                    )
                    
                    print(log_path)

                    model.load_weights(log_path)
                    print(field_preds.shape)
                    # print(field_ys.shape)
                    predictions = model.predict(np.reshape(field_preds, (-1,4,240, 240)))
                    # predictions = np.reshape(field_preds, (preds.shape[0], 4, 4))
                    if unlabeled == "":
                        y = np.reshape(field_ys, (-1,4,240, 240))

        except AttributeError as e:
            print(e)

    if unlabeled == "":
        if problem == "detection":
            ACC, TPR, TNR, ROC_AUC = SkillScores.get_skill_scores(predictions, y)
        else:
            print("predictions, y")
            print(predictions.shape)
            print(y.shape)
            ACC, TPR, TNR, ROC_AUC, dice, fpr, tpr = SkillScores.get_skill_scores_localization(
                predictions, y
            )

    all_files = []
    for file in filenames:
        all_files.append([file] * 25)
    filenames = np.array(all_files)

    print("PREDICTIONS")
    print(predictions.shape)
    # print("GROUND TRUTH")
    # print(y.shape)
    print("FILENAMES")
    print(filenames.shape)

    print("unlabeled")
    print(unlabeled)
    if unlabeled == "":
        if problem == "detection":
            SkillScores.print_skill_scores(ACC, TPR, TNR, ROC_AUC)
        else:
            SkillScores.print_skill_scores(ACC, TPR, TNR, ROC_AUC, dice)

    if unlabeled == "":
        if problem == "detection":
            with open(
                "skill_scores" + model_file + str(loadfile) + ".csv", mode="w"
            ) as predict_file:
                writer = csv.writer(predict_file, delimiter=",")
                writer.writerow(["ACC", "TPR", "TNR", "ROC_AUC"])
                writer.writerow([ACC, TPR, TNR, ROC_AUC])
        else:
            with open(
                "skill_scores_localization_" + model_file + str(loadfile) + ".csv",
                mode="w",
            ) as predict_file:
                writer = csv.writer(predict_file, delimiter=",")
                writer.writerow(["ACC", "TPR", "TNR", "ROC_AUC", "Dice", "fpr", "tpr"])
                writer.writerow([ACC, TPR, TNR, ROC_AUC, dice, fpr, tpr])

    if problem == "detection":
        if unlabeled == "":
            with open(
                "true_predictions_" + model_file + str(loadfile) + ".csv", mode="w"
            ) as predict_file:
                writer = csv.writer(predict_file, delimiter=",")
                print(np.array(filenames).shape)
                print(np.array(y).shape)
                print(np.array(predictions).shape)
                for i in range(len(predictions)):
                    writer.writerow([filenames[i][0], y[i][0], predictions[i][0]])
        else:
            with open(
                "true_predictions_2019_" + model_file + str(loadfile) + ".csv", mode="w"
            ) as predict_file:
                writer = csv.writer(predict_file, delimiter=",")
                print(np.array(filenames).shape)
                print(np.array(predictions).shape)
                for i in range(len(predictions)):
                    writer.writerow([filenames[i][0], predictions[i][0]])
    else:
        print(filenames.shape)
        # print(y.shape)
        print(predictions.shape)
        if model_name == utils.ML_Model.Shallow_CNN_All:
            predictions = np.reshape(
                predictions,
                (
                    predictions.shape[0],
                    predictions.shape[2],
                    predictions.shape[3],
                    1,
                ),
            )
            if unlabeled == "":
                y = np.reshape(y, (4, y.shape[0], y.shape[2], y.shape[3]))[0]
                

        for i in range(len(filenames)):
            if unlabeled == "":
                cv2.imwrite(
                    settings.WORKING_DIRECTORY
                    + "localization_preds_"
                    + model_file
                    + "/"
                    + filenames[i][0]
                    + ".png",
                    predictions[i],
                )
            else:
                print(settings.WORKING_DIRECTORY+ "localization_preds_2019_"+ model_file+ "/"+ filenames[i][0]+ ".png")
                cv2.imwrite(
                    settings.WORKING_DIRECTORY
                    + "localization_preds_2019_"
                    + model_file
                    + "/"
                    + filenames[i][0]
                    + ".png",
                    predictions[i],
                )
                print(settings.WORKING_DIRECTORY+ "localization_preds_2019_"+ model_file+ "/"+ filenames[    i][0]+ ".png")

            if unlabeled == "":
                cv2.imwrite(
                    settings.WORKING_DIRECTORY
                    + "localization_truth_"
                    + model_file
                    + "/"
                    + filenames[i][0]
                    + ".png",
                    y[i],
                )

    if model_name == utils.ML_Model.Shallow_CNN and unlabeled == "":
        if problem == "detection":
            loss, acc = model.evaluate(x, y)
            print("LOSS, ACC: ")
            print(loss, acc)
        else:
            print(x.shape)
            print(y.shape)
            loss, metric = model.evaluate(x, y)
            print("LOSS, METRIC: ")
            print(loss, metric)


def main(results):
    os.chdir(settings.WORKING_DIRECTORY)
    radar_product = utils.Radar_Products(results.radar_product)
    if results.log_path is None:
        c = radar_product.fullname
        print(c)
        a = ml_utils.LOG_PATH.format(radar_product.fullname)
        b = ml_utils.KERAS_SAVE_FILE.format(radar_product.fullname, "")
        log_path = os.path.join(
            ml_utils.LOG_PATH.format(radar_product.fullname),
            ml_utils.KERAS_SAVE_FILE.format(radar_product.fullname, ""),
        )
    else:
        log_path = results.log_path

    print(log_path)

    model = utils.ML_Model(results.model)

    eval(
        log_path=log_path,
        radar_product=radar_product,
        coord_conv=results.coord_conv,
        problem=results.problem,
        dual_pol=results.dual_pol,
        num_temporal_data=results.num_temporal_data,
        model_name=model,
        loadfile=results.loadfile,
        unlabeled=results.unlabeled,
    )


if __name__ == "__main__":
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
        default="Velocity.h5",
        help="""
        Optionally input the location of the save file where the default is
        model/radar_product/radar_product.h5
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
        "-m",
        "--model",
        type=int,
        default=0,
        help="""
            Use an integer to select a model from the following list:
                0 : Shallow CNN
                1 : Shallow CNN, all radar products
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
    parser.add_argument(
        "-ul",
        "--unlabeled",
        type=str,
        default="",
        help="""
        path to unlabeled dataset (optional)
        """,
    )
    results = parser.parse_args()
    main(results)
