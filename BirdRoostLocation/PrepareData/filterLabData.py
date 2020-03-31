import pandas
import os
from glob import glob
import shutil


def filter_data():
    lab_output_csv = "/Users/Kate/workspace/BirdRoostLocation/MLData/new_data_filtered/no_preds/output.csv"
    original_data = (
        "/Users/Kate/workspace/BirdRoostLocation/MLData/new_data_filtered/no_preds/"
    )
    fields = ["Roost_Reflectivity", "Roost_Velocity", "Roost_Rho_HV", "Roost_Zdr"]
    filtered_data = "/Users/Kate/workspace/BirdRoostLocation/MLData/no_rings_filtered/"

    colnames = ["filename", "latitude", "longitude", "flag"]
    data = pandas.read_csv(lab_output_csv, names=colnames)

    filenames = data.filename.tolist()
    counter = 0

    print(len(filenames))
    data2 = pandas.read_csv(lab_output_csv, names=colnames)
    print(len(data.filename.tolist()))
    for filename in filenames:
        for field in fields:
            # print(str(filename)[0:10])
            for file in glob(
                original_data + str(field) + "/??" + str(filename)[0:10] + "*"
            ):
                if field == "Roost_Reflectivity":
                    counter += 1
                    print(counter)
                shutil.move(file, filtered_data + str(field) + "/")


if __name__ == "__main__":
    filter_data()

