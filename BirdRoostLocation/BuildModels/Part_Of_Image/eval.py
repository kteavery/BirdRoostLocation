import argparse
import os
import BirdRoostLocation.BuildModels.ShallowCNN.model as ml_model
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation import utils
from BirdRoostLocation.ReadData import BatchGenerator
from BirdRoostLocation.Analysis import skill_scores


def eval(log_path, radar_product):
    batch_generator = BatchGenerator.Small_Image_Batch_Generator(
        ml_label_csv=settings.LABEL_CSV,
        ml_split_csv=settings.ML_SPLITS_DATA,
        high_memory_mode=False,
    )

    x, y, filenames = batch_generator.get_batch(
        ml_set=utils.ML_Set.testing, dualPol=False, radar_product=radar_product
    )

    model = ml_model.smaller_build_model(inputDimensions=(80, 80, 1))
    model.load_weights(log_path)

    predictions = model.predict(x)
    ACC, TPR, TNR, ROC_AUC = skill_scores.get_skill_scores(predictions[:, 0], y[:, 0])
    skill_scores.print_skill_scores(ACC, TPR, TNR, ROC_AUC)


def main(results):
    os.chdir(settings.WORKING_DIRECTORY)
    radar_product = utils.Radar_Products(results.radar_product)
    log_path = "model/small_images/Reflectivity.h5"

    print(log_path)
    eval(log_path=log_path, radar_product=radar_product)


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

    results = parser.parse_args()
    main(results)
