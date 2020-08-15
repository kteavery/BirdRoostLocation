import pandas
import os
from glob import glob
import shutil
import BirdRoostLocation.LoadSettings as settings


def filter_data():
    lab_output_csv = (
        settings.WORKING_DIRECTORY + "new_data_filtered/no_preds/output.csv"
    )
    original_data = settings.WORKING_DIRECTORY + "new_data_filtered/no_preds/"
    fields = ["Roost_Reflectivity", "Roost_Velocity", "Roost_Rho_HV", "Roost_Zdr"]
    filtered_data = settings.WORKING_DIRECTORY + "no_rings_filtered/"

    colnames = ["filename", "latitude", "longitude", "flag"]
    data = pandas.read_csv(lab_output_csv, names=colnames)

    filenames = data.filename.tolist()
    counter = 0

    print(len(filenames))
    data2 = pandas.read_csv(lab_output_csv, names=colnames)
    print(len(data.filename.tolist()))
    for filename in filenames:
        for field in fields:
            for file in glob(
                original_data + str(field) + "/??" + str(filename)[0:10] + "*"
            ):
                if field == "Roost_Reflectivity":
                    counter += 1
                    print(counter)
                shutil.move(file, filtered_data + str(field) + "/")


if __name__ == "__main__":
    filter_data()
