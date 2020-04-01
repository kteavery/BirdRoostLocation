import matplotlib.pyplot as plt
import math
import pandas
import pylab as pl
import numpy as np
import re
import csv, os
import pyart
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.PrepareData import VisualizeNexradData


def visualizeResults(image, truth, prediction, path):
    """
    images: array of images to plot (1, 2, or 4 images)
    truth: tuple of (theta, radius)
    prediction: tuple of (theta, radius)
    """
    data = plt.imread(image)

    fig = plt.figure()

    # create polar axes in the foreground and remove its background
    # to see through
    ax = fig.add_subplot(111, polar=True, label="polar", zorder=1)
    ax.set_facecolor("None")
    ax.scatter(truth[0], truth[1], c="black", marker="*", s=70.0)
    circle100 = plt.Circle(
        (0, 0), 100, transform=ax.transData._b, color="black", fill=False, linewidth=1.0
    )
    ax.add_artist(circle100)
    circle200 = plt.Circle(
        (0, 0), 200, transform=ax.transData._b, color="black", fill=False, linewidth=1.0
    )
    ax.add_artist(circle200)
    circle300 = plt.Circle(
        (0, 0), 300, transform=ax.transData._b, color="black", fill=False, linewidth=1.0
    )
    ax.add_artist(circle300)
    ax.scatter(prediction[0], prediction[1], c="black", marker="o")
    ax.axis("off")
    ax.set_rmax(300)

    # create axes in the background to show cartesian image
    ax0 = fig.add_subplot(111, zorder=0)
    ax0.imshow(data)  # , extent=[0, 1, 0, 1])

    plt.show()
    # plt.savefig(path)

    plt.close()


if __name__ == "__main__":
    df = pandas.read_csv(settings.WORKING_DIRECTORY + "/" + settings.LABEL_CSV)
    df = df.loc[df["radar"] == "KLIX"].reset_index()
    print(df.head())
    fields = ["Reflectivity", "Velocity", "Rho_HV", "Zdr"]

    for i in range(len(df)):
        aws_file = df["AWS_file"].iloc[[i]]
        theta = df["polar_theta"].iloc[[i]]
        radius = df["polar_radius"].iloc[[i]]

        for field in fields:
            full_path = (
                "/Users/Kate/workspace/BirdRoostLocation/MLData/no_rings_filtered/Roost_"
                + field
                + "/"
                # + aws_file[i][10:12]
                + aws_file[i]
                + "_"
                + field
                + ".png"
            )
            save_path = (
                "/Users/Kate/workspace/BirdRoostLocation/MLData/all_true_data/Roost_"
                + field
                + "/"
                # + aws_file[i][10:12]
                + aws_file[i]
                + "_"
                + field
                + ".png"
            )
            try:
                print(settings.WORKING_DIRECTORY + "/" + settings.LABEL_CSV)
                print((math.radians(theta[i] + 180), radius[i]))
                print((theta[i], radius[i]))
                print(full_path)
                if os.path.isfile(full_path):
                    visualizeResults(
                        full_path,
                        # (math.radians(float(truth[1])), float(truth[0])),
                        (math.radians(theta[i] + 180), radius[i]),
                        (0, 0),
                        save_path,
                    )
            except:
                print(aws_file[i] + " was passed")
