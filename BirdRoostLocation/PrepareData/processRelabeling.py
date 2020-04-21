import pandas as pd
import math
import numpy as np
import copy

from BirdRoostLocation import LoadSettings as settings

X = 230  # pixel location of center
# X = 231
Y = 230
KMLAT = 110.574  # km = 1 degree latitude
MULTLONG = 111.320  # multiplier


def convertLatLong(labels):
    latLongLabels = copy.deepcopy(labels)
    nexrads = pd.read_csv(
        settings.WORKING_DIRECTORY + "/nexrad.csv",
        names=["radar", "latitude", "longitude"],
        header=None,
    )
    for radar in nexrads["radar"]:
        latPx = latLongLabels.loc[latLongLabels.filename.str.match(radar), "latitude"]
        latKm = (latPx - X) * (300 / (X * 2))
        # latKm = latPx * (300 / (X * 2))
        longPx = latLongLabels.loc[latLongLabels.filename.str.match(radar), "longitude"]
        longKm = (longPx - Y) * (300 / (Y * 2))
        # longKm = longPx * (300 / (Y * 2))

        newLat = latKm * (1 / KMLAT)
        newLong = longKm * (1 / (MULTLONG * (newLat * (math.pi / 180)).apply(math.cos)))

        if radar == "KFSD":
            print("latPx")
            print(latPx)
            print(300 / (Y * 2))
            # print("latKm")
            # print(latKm)
            print("longPx")
            print(longPx)
            print(300 / (Y * 2))
            # print("longKm")
            # print(longKm)
            print("newLat")
            print(newLat)
            print("newLong")
            print(newLong)
            print("newLat + nexrads.loc[nexrads['radar'] == radar]['latitude'].item()")
            print(newLat + nexrads.loc[nexrads["radar"] == radar]["latitude"].item())
            print(
                "newLong + nexrads.loc[nexrads['radar'] == radar]['longitude'].item()"
            )
            print(newLong + nexrads.loc[nexrads["radar"] == radar]["longitude"].item())

        latLongLabels.loc[latLongLabels.filename.str.match(radar), "latitude"] = (
            newLat + nexrads.loc[nexrads["radar"] == radar]["latitude"].item()
        )
        latLongLabels.loc[latLongLabels.filename.str.match(radar), "longitude"] = (
            newLong + nexrads.loc[nexrads["radar"] == radar]["longitude"].item()
        )

    # print(latLongLabels)

    return latLongLabels


def combineN(inputDF):
    df = copy.deepcopy(inputDF)
    # df = df.reset_index(drop=True)
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
                    # print(str(df["latitude"][i]) + ", " + str(df["longitude"][i]))
                    # print(str(df["latitude"][j]) + ", " + str(df["longitude"][j]))
                    lat = (df["latitude"][i] + df["latitude"][j]) / 2
                    long = (df["longitude"][i] + df["longitude"][j]) / 2
                    df["latitude"][i] = lat
                    df["longitude"][i] = long
                    df = df.drop(j)
                except KeyError:
                    continue
    # print(inputDF.head())
    # print(df.head())

    return df


def processLabels(labels):
    """
    1. combine all clicks within N pixels
    2. take all false flags as roosts
    3. a) throw away all true flagged labels b) enough true flags over a period
    equal a false flag
    """

    print(labels.head())
    falses = labels.groupby(["flag"]).get_group(False)
    trues = labels.groupby(["flag"]).get_group(True)

    # print(falses.head())

    newLabels = falses.groupby(falses["filename"].str[:12]).apply(combineN)
    # print(newLabels)

    return newLabels


def copySameLabels(labels):
    labelDF = copy.deepcopy(labels[["filename", "latitude", "longitude", "flag"]])
    print(len(labelDF))
    timestamps = pd.read_csv(settings.WORKING_DIRECTORY + "/true_ml_labels_polar.csv")[
        "AWS_file"
    ]

    for index, label in labels[
        ["filename", "latitude", "longitude", "flag"]
    ].iterrows():
        sharedStamps = timestamps[
            timestamps.str.contains(label["filename"][0:10])
        ].to_frame()

        sharedStamps.columns = ["AWS_file"]
        sharedStamps["latitude"] = label["latitude"]
        sharedStamps["longitude"] = label["longitude"]
        sharedStamps["flag"] = label["flag"]
        # print(type(sharedStamps))
        # labelDF.append(sharedStamps, ignore_index=True)
        labelDF = pd.concat([labelDF, sharedStamps], axis=0, sort=True)

    print(len(labelDF))
    return labelDF


def main():
    labels = pd.read_csv(settings.WORKING_DIRECTORY + "/all_true_data/output.csv")
    newLabels = processLabels(labels)
    latLongLabels = convertLatLong(newLabels)
    extendedLabels = copySameLabels(latLongLabels)
    extendedLabels[["AWS_file", "latitude", "longitude", "flag"]].to_csv(
        settings.WORKING_DIRECTORY + "/processed_relabels.csv", index=False
    )


if __name__ == "__main__":
    main()
