import pandas
import os
from glob import glob
import shutil


def filter_data():
    lab_output_csv = (
        "/Users/Kate/workspace/BirdRoostLocation/MLData/app_outputs/lab_output.csv"
    )
    original_data = "/Users/Kate/workspace/BirdRoostLocation/MLData/no_rings/"
    fields = ["Roost_Reflectivity", "Roost_Velocity", "Roost_Rho_HV", "Roost_Zdr"]
    filtered_data = "/Users/Kate/workspace/BirdRoostLocation/MLData/all_true_data/"

    colnames = ["filename", "latitude", "longitude", "flag"]
    data = pandas.read_csv(lab_output_csv, names=colnames)

    filenames = data.filename.tolist()

    for filename in filenames:
        for field in fields:
            for file in glob(original_data + str(field) + "/??" + str(filename) + "*"):
                # print(file)
                shutil.move(file, filtered_data + str(field) + "/")


if __name__ == "__main__":
    filter_data()

