import pandas as pd
import math
import numpy as np
import copy

from BirdRoostLocation import LoadSettings as settings

X = 231  # pixel location of center
Y = 230
KMLAT = 110.574  # km = 1 degree latitude
MULTLONG = 111.320  # multiplier
PXTOKM = 300 / 174  # km = 1 px


def convertLatLong(labels):
    latLongLabels = copy.deepcopy(labels)
    nexrads = pd.read_csv(
        settings.WORKING_DIRECTORY + "/nexrad.csv",
        names=["radar", "latitude", "longitude"],
        header=None,
    )
    for radar in nexrads["radar"]:
        kmOrigin = 300 * math.sqrt(2)
        latPx = latLongLabels.loc[latLongLabels.filename.str.match(radar), "latitude"]
        latKm = (X - latPx) * (300 / 174)

        longPx = latLongLabels.loc[latLongLabels.filename.str.match(radar), "longitude"]
        longKm = (longPx - Y) * (300 / 174)

        newLat = latKm * (1 / KMLAT)
        newLong = longKm * (1 / (MULTLONG * (newLat * (math.pi / 180)).apply(math.cos)))

        latLongLabels.loc[latLongLabels.filename.str.match(radar), "latitude"] = (
            newLat + nexrads.loc[nexrads["radar"] == radar]["latitude"].item()
        )
        latLongLabels.loc[latLongLabels.filename.str.match(radar), "longitude"] = (
            newLong + nexrads.loc[nexrads["radar"] == radar]["longitude"].item()
        )

    print(latLongLabels)

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

    falses = labels.groupby(["flag"]).get_group(False)
    trues = labels.groupby(["flag"]).get_group(True)

    # print(falses.head())

    newLabels = falses.groupby(["filename"]).apply(combineN)
    # print(newLabels)

    return newLabels


def main():
    labels = pd.read_csv(settings.WORKING_DIRECTORY + "/all_true_data/output.csv")
    newLabels = processLabels(labels)
    latLongLabels = convertLatLong(newLabels)

    # write latLongLabels to csv


if __name__ == "__main__":
    main()

