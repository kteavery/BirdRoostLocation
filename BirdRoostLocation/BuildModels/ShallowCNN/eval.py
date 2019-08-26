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

import BirdRoostLocation.BuildModels.ShallowCNN.model as ml_model
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation import utils
from BirdRoostLocation.BuildModels import ml_utils
from BirdRoostLocation.ReadData import BatchGenerator


def eval(log_path, radar_product):
    """Evaluate the shallow CNN model trained on a single radar product.

        Args:
            log_path: The location of the save directory. This method will
                read the save located in this directory.
            radar_product: The radar product the model is evaluating. This
                should be a value of type utils.Radar_Products.

        """
    batch_generator = BatchGenerator.Batch_Generator(
        ml_label_csv=settings.LABEL_CSV,
        ml_split_csv=settings.ML_SPLITS_DATA,
        validate_k_index=3,
        test_k_index=4,
        default_batch_size=5000,
    )

    x, y, filenames = batch_generator.get_batch(utils.ML_Set.testing, radar_product)
    model = ml_model.build_model(inputDimensions=(240, 240, 3))
    model.load_weights(log_path)

    loss, acc = model.evaluate(x, y)
    print(loss, acc)


def main(results):
    os.chdir(settings.WORKING_DIRECTORY)
    radar_product = utils.Radar_Products(results.radar_product)
    if results.log_path is None:
        log_path = os.path.join(
            ml_utils.LOG_PATH.format(radar_product.fullname),
            ml_utils.KERAS_SAVE_FILE.format(radar_product.fullname, ""),
        )
    else:
        log_path = results.log_path

    print(log_path)
    eval(log_path=log_path, radar_product=radar_product)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--radar_product",
        type=int,
        default=0,
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
        model/radar_product/radar_product.h5
        """,
    )
    results = parser.parse_args()
    main(results)
