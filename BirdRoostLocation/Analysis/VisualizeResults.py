import matplotlib.pyplot as plt
import math
import pandas
import csv, os
import pyart
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.PrepareData import VisualizeNexradData


def visualizeResults(images, truth, prediction):
    """
    images: array of images to plot (1, 2, or 4 images)
    truth: tuple of (theta, radius)
    prediction: tuple of (theta, radius)
    """
    data = plt.imread(images[0])

    fig = plt.figure()
    # create axes in the background to show cartesian image
    ax0 = fig.add_subplot(111)
    # ax0.axis("off")
    ax0.imshow(data)

    # create polar axes in the foreground and remove its background
    # to see through
    ax = fig.add_subplot(111, polar=True, label="polar")
    ax.set_facecolor("None")
    ax.scatter(truth[0], truth[1])
    ax.scatter(0, 0)
    ax.axis("off")
    ax.set_rmax(300)

    plt.show()


if __name__ == "__main__":
    df = pandas.read_csv(settings.WORKING_DIRECTORY + "/" + settings.LABEL_CSV)

    row = df.loc[df["AWS_file"] == "KFCX20140630_101751_V06"].iloc[0]

    print(row["polar_radius"])
    print(math.radians(row["polar_theta"]))
    print(row["polar_theta"])
    print()
    visualizeResults(
        [
            "/Users/Kate/workspace/BirdRoostLocation/MLData/data/Roost_Reflectivity/30KFCX20140630_101751_V06_Reflectivity.png"
        ],
        (math.radians(row["polar_theta"]), row["polar_radius"]),
        (0, 0),
    )

    row = df.loc[df["AWS_file"] == "KCAE20140711_103216_V06"].iloc[0]
    print(row["polar_radius"])
    print(math.radians(row["polar_theta"]))
    print(row["polar_theta"])
    print()
    visualizeResults(
        [
            "/Users/Kate/workspace/BirdRoostLocation/MLData/data/Roost_Reflectivity/11KCAE20140711_103216_V06_Reflectivity.png"
        ],
        (math.radians(row["polar_theta"]), row["polar_radius"]),
        (0, 0),
    )

    row = df.loc[df["AWS_file"] == "KLWX20170722_100914_V06"].iloc[0]
    print(row["polar_radius"])
    print(math.radians(row["polar_theta"]))
    print(row["polar_theta"])
    print()
    visualizeResults(
        [
            "/Users/Kate/workspace/BirdRoostLocation/MLData/data/Roost_Reflectivity/22KLWX20170722_100914_V06_Reflectivity.png"
        ],
        (math.radians(row["polar_theta"]), row["polar_radius"]),
        (0, 0),
    )

