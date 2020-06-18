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
import pandas as pd
from ast import literal_eval

import BirdRoostLocation.BuildModels.ShallowCNN.model as ml_model
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation import utils
from BirdRoostLocation.BuildModels import ml_utils
from BirdRoostLocation.ReadData import BatchGenerator
from BirdRoostLocation.Analysis import SkillScores
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
import keras


def field_predict(x, log_path, coord_conv, problem):
    model = ml_model.build_model(
        inputDimensions=(240, 240, 3), coord_conv=coord_conv, problem=problem
    )

    model.load_weights(log_path)
    predictions = model.predict(x)

    return predictions


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
    while type(x) == type(None) and type(y) == type(None):
        x, y, filenames = batch_generator.get_batch(
            utils.ML_Set.testing,
            dualPol=dual_pol,
            radar_product=radar_product,
            num_temporal_data=num_temporal_data,
            problem=problem,
        )

        try:
            print(x.shape)
            print(y.shape)
            print(filenames.shape)
        except AttributeError as e:
            print(e)

    if model_name == utils.ML_Model.Shallow_CNN:
        predictions = field_predict(x, log_path, coord_conv, problem)

    else:
        agg_model = Sequential()
        agg_model.add(Dense(256, input_shape=(4, 2), activation="relu"))
        agg_model.add(Dense(2, activation="softmax"))
        agg_model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.adam(lr),
            metrics=["accuracy"],
        )

        agg_model.load_weights(log_path)

        field_ys = np.array([])
        field_preds = np.array([])
        for field in ["Reflectivity", "Velocity", "Rho_HV", "Zdr"]:
            print(field)
            preds = field_predict(
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

            preds = np.array([np.array([j, 1.0 - j]) for j in preds])
            field_y = np.array([np.array([j, 1.0 - j]) for j in y])

            print(preds.shape)
            print(field_y.shape)

            field_preds = np.append(field_preds, preds)
            field_ys = np.append(field_ys, field_y)

        print(field_preds.shape)
        print(field_ys.shape)
        predictions = np.reshape(field_preds, (preds.shape[0], 4, 4))
        y = np.reshape(field_ys, (preds.shape[0], 4, 4))

    ACC, TPR, TNR, ROC_AUC = SkillScores.get_skill_scores(predictions, y)

    # ACC_RAD = SkillScores.get_skill_scores_regression(predictions[:, 0], y[:, 0], 0.1)
    # print("ACC_RAD: " + str(ACC_RAD))

    # ACC_THETA = SkillScores.get_skill_scores_regression(predictions[:, 1], y[:, 1], 0.1)
    # print("ACC_THETA: " + str(ACC_THETA))

    print("PREDICTIONS")
    print(len(predictions))
    print("GROUND TRUTH")
    print(len(y))
    print("FILENAMES")
    print(len(filenames))

    SkillScores.print_skill_scores(ACC, TPR, TNR, ROC_AUC)

    with open(
        "skill_scores" + model_file + str(loadfile) + ".csv", mode="w"
    ) as predict_file:
        writer = csv.writer(predict_file, delimiter=",")
        writer.writerow(["ACC", "TPR", "TNR", "ROC_AUC"])
        writer.writerow([ACC, TPR, TNR, ROC_AUC])

    with open(
        "true_predictions_" + model_file + str(loadfile) + ".csv", mode="w"
    ) as predict_file:
        writer = csv.writer(predict_file, delimiter=",")
        print(np.array(filenames).shape)
        print(np.array(y).shape)
        print(np.array(predictions).shape)
        for i in range(len(predictions)):
            writer.writerow([filenames[i], y[i][0], predictions[i][0]])

    if model_name == utils.ML_Model.Shallow_CNN:
        if problem == "detection":
            loss, acc = model.evaluate(x, y)
            print("LOSS, ACC: ")
            print(loss, acc)
        else:
            loss, mae, mape, cosine = model.evaluate(x, y)
            print("LOSS, MAE, MAPE, COSINE: ")
            print(loss, mae, mape, cosine)


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
    results = parser.parse_args()
    main(results)
