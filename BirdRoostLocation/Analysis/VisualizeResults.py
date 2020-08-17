import matplotlib.pyplot as plt
import math, sys
import pandas
from PIL import Image
import pylab as pl
import numpy as np
import re
import csv, os
import pyart
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.PrepareData import VisualizeNexradData


def points_in_circle_np(radius, y0=0, x0=0):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    y, x = np.where((y_[:, np.newaxis] - y0) ** 2 + (x_ - x0) ** 2 <= radius ** 2)
    for y, x in zip(y_[y], x_[x]):
        yield y, x


def visualizeMask(truth):
    mask = np.zeros((240, 240))

    for roost in truth:
        try:
            mask_roost_size = (roost[1] / 300) * (240 / 2)

            cartx = mask_roost_size * math.cos(roost[0])
            carty = mask_roost_size * math.sin(roost[0])

            mask[120 - int(round(carty)), 120 + int(round(cartx))] = 1.0

            print(str(120 - int(round(carty))) + ", " + str(120 + int(round(cartx))))

            color_pts = points_in_circle_np(
                28.0, y0=120 - int(round(carty)), x0=120 + int(round(cartx))
            )

            for pt in color_pts:
                mask[pt[0], pt[1]] = 1.0
        except IndexError as e:
            pass

    np.set_printoptions(threshold=sys.maxsize)
    img = Image.fromarray((mask * 255).astype("uint8"), "L")
    img.show()


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

    circle = plt.Circle(
        (truth[1] * math.cos(truth[0]), truth[1] * math.sin(truth[0])),
        28.0,
        transform=ax.transData._b,
        color="black",
        alpha=0.5,
    )
    ax.add_artist(circle)

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
    ax.axis("off")
    ax.set_rmax(300)

    # create axes in the background to show cartesian image
    ax0 = fig.add_subplot(111, zorder=0)
    ax0.imshow(data)

    plt.show()

    plt.close()


if __name__ == "__main__":
    df = pandas.read_csv(settings.WORKING_DIRECTORY + "/" + settings.LABEL_CSV)
    df = df.loc[df["radar"] == "KLIX"].reset_index()
    print(df.head())
    fields = ["Reflectivity", "Velocity", "Rho_HV", "Zdr"]

    suffixes = [
        "",
        "_flip",
        "_flip_noise",
        "_noise",
        "_45_noise",
        "_45",
        "_flip_45",
        "_90_noise",
        "_90",
        "_flip_90",
        "_135_noise",
        "_135",
        "_flip_135",
        "_180_noise",
        "_180",
        "_flip_180",
        "_225_noise",
        "_225",
        "_flip_225",
        "_270_noise",
        "_270",
        "_flip_270",
        "_315_noise",
        "_315",
        "_flip_315",
    ]

    visualizeMask([(math.radians(-2.0), 172.0)])

