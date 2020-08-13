import pandas as pd
import math
import numpy as np
import copy

from BirdRoostLocation import LoadSettings as settings

X = 230  # pixel location of center
Y = 230
KMLAT = 110.574  # km = 1 degree latitude
MULTLONG = 111.320  # multiplier


def combineN(inputDF):
    df = copy.deepcopy(inputDF)
    N = 20
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i < j and (
                math.sqrt(
                    (row1["latitude"] - row2["latitude"]) ** 2
                    + (row1["longitude"] - row2["longitude"]) ** 2
                )
                < N
            ):
                try:
                    print(str(df["latitude"][i]) + ", " + str(df["longitude"][i]))
                    print(str(df["latitude"][j]) + ", " + str(df["longitude"][j]))
                    lat = (df["latitude"][i] + df["latitude"][j]) / 2
                    long = (df["longitude"][i] + df["longitude"][j]) / 2
                    df["latitude"][i] = lat
                    df["longitude"][i] = long
                    df = df.drop(j)
                except KeyError as e:
                    print(e)

    return df


def processLabels(labels):
    """
    1. combine all clicks within N pixels
    2. take all false flags as roosts
    3. a) throw away all true flagged labels b) enough true flags over a period
    equal a false flag
    """

    print(labels.head())
    trues = labels.groupby(["Roost"]).get_group(True)
    newLabels = combineN(trues)
    print(newLabels.head())

    return newLabels


def copySameLabels(labels):
    labelDF = copy.deepcopy(labels)
    print(len(labelDF))
    timestamps = pd.read_csv(settings.WORKING_DIRECTORY + "/true_ml_labels_polar.csv")[
        "AWS_file"
    ]

    for index, label in labels.iterrows():
        sharedStamps = timestamps[
            timestamps.str.contains(label["AWS_file"][0:10])
        ].to_frame()

        sharedStamps.columns = ["AWS_file"]
        sharedStamps["latitude"] = label["latitude"]
        sharedStamps["longitude"] = label["longitude"]
        sharedStamps["Roost"] = label["Roost"]
        sharedStamps["label_origin"] = label["label_origin"]
        sharedStamps["nexrad_lat"] = label["nexrad_lat"]
        sharedStamps["nexrad_lon"] = label["nexrad_lon"]
        sharedStamps["polar_radius"] = label["polar_radius"]
        sharedStamps["polar_theta"] = label["polar_theta"]
        sharedStamps["radar"] = label["radar"]
        sharedStamps["radius"] = label["radius"]
        sharedStamps["roost_id"] = label["roost_id"]
        sharedStamps["roost_time"] = label["roost_time"]
        sharedStamps["sunrise_time"] = label["sunrise_time"]

        labelDF = pd.concat([labelDF, sharedStamps], axis=0, sort=True)

    print(len(labelDF))
    return labelDF


def main():
    labels = pd.read_csv(settings.WORKING_DIRECTORY + "true_ml_relabels_polar.csv")
    lables = labels.sort_values(by=["AWS_file"])
    extendedLabels = copy.deepcopy(labels)
    extendedLabels.to_csv(
        settings.WORKING_DIRECTORY + "/true_ml_relabels_polar_short.csv", index=False
    )


if __name__ == "__main__":
    main()
