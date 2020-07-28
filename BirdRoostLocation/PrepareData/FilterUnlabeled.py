import BirdRoostLocation.LoadSettings as settings
import pandas as pd
import shutil, os


def filter_unlabeled(csv, src, dest):
    data = pd.read_csv(csv)
    data_true = data[data["predictions"] >= 0.5]
    data_true_names = data_true["filenames"]

    for f in data_true_names:
        shutil.copy(
            src + "/Reflectivity_Color/" + f[0:19] + "_V06_Reflectivity.png",
            dest + "/Reflectivity_Color/",
        )
        shutil.copy(
            src + "/Velocity_Color/" + f[0:19] + "_V06_Velocity.png",
            dest + "/Velocity_Color/",
        )
        shutil.copy(
            src + "/Rho_HV_Color/" + f[0:19] + "_V06_Rho_HV.png", dest + "/Rho_HV_Color/"
        )
        shutil.copy(src + "/Zdr_Color/" + f[0:19] + "_V06_Zdr.png", dest + "/Zdr_Color/")


if __name__ == "__main__":
    filter_unlabeled(
        settings.WORKING_DIRECTORY + "true_predictions_2019_Zdr2.csv",
        settings.WORKING_DIRECTORY + "2019images",
        settings.WORKING_DIRECTORY + "2019images_filtered",
    )

