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
    # ax.scatter(truth[0], truth[1], c="black", marker="*", s=70.0)

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
    # ax.scatter(prediction[0], prediction[1], c="black", marker="o")
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

    # KEVXtheta = [
    #     219.04207776024253,
    #     320.95792223975747,
    #     320.95792223975747,
    #     219.04207776024253,
    #     264.04207776024253,
    #     264.04207776024253,
    #     365.95792223975747,
    #     309.04207776024253,
    #     309.04207776024253,
    #     410.95792223975747,
    #     354.04207776024253,
    #     354.04207776024253,
    #     455.95792223975747,
    #     399.04207776024253,
    #     399.04207776024253,
    #     500.95792223975747,
    #     444.04207776024253,
    #     444.04207776024253,
    #     545.9579222397574,
    #     489.04207776024253,
    #     489.04207776024253,
    #     590.9579222397574,
    #     534.0420777602426,
    #     534.0420777602426,
    #     635.9579222397574,
    # ]

    # KEVXradius = [0.10477419878244053] * len(KEVXradius)

    # labels = pandas.read_csv(
    #     "/Users/Kate/workspace/BirdRoostLocation/MLData/true_ml_relabels_polar_short.csv"
    # )
    # KEVXrows = labels.loc[labels["AWS_file"] == "KEVX20130724_110326_V06"]
    # points = [
    #     (math.radians(float(row["polar_theta"])), float(row["polar_radius"]))
    #     for index, row in KEVXrows.iterrows()
    # ]

    # for i, suffix in enumerate(suffixes):
    visualizeMask(
        [
            (math.radians(-170.64540477317283), 117.24481269235102),
            (math.radians(22.59922891088621), 45.08385832876151),
            (math.radians(-117.54880097214688), 209.08349349634045),
            (math.radians(27.962537078732918), 77.59194126687083),
            (math.radians(69.36764465370331), 44.429575689160735),
            (math.radians(24.688976464036585), 62.21669720022375),
            (math.radians(-167.3837851418881), 111.04753441365965),
            (math.radians(-131.0324743950738), 135.50595496468856),
            (math.radians(-146.13241503834084), 84.09156632237392),
        ]
    )

    # visualizeResults(
    #     "/Users/Kate/workspace/BirdRoostLocation/MLData/KEVX20130724_110326_V06/24KEVX20130724_110326_V06_Reflectivity"
    #     + suffix
    #     + ".png",
    #     (math.radians(points[i]), points[i]),
    #     (0, 0),
    #     "/Users/Kate/workspace/BirdRoostLocation/MLData/KEVX20130724_110326_V06/24KEVX20130724_110326_V06_Reflectivity"
    #     + suffix
    #     + ".png",
    # )
