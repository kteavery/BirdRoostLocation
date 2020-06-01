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
    mask_roost_size = (truth[1] / 300) * (240 / 2)

    cartx = mask_roost_size * math.cos(truth[0])
    carty = mask_roost_size * math.sin(truth[0])

    mask[120 - int(round(carty)), 120 + int(round(cartx))] = 1.0

    print(str(120 - int(round(carty))) + ", " + str(120 + int(round(cartx))))

    color_pts = points_in_circle_np(
        mask_roost_size, y0=120 - int(round(carty)), x0=120 + int(round(cartx))
    )

    for pt in color_pts:
        mask[pt[0], pt[1]] = 1.0

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

    KEVXradius = [
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
        0.10477419878244053,
    ]
    KEVXtheta = [
        219.04207776024253,
        320.95792223975747,
        320.95792223975747,
        219.04207776024253,
        264.04207776024253,
        264.04207776024253,
        365.95792223975747,
        309.04207776024253,
        309.04207776024253,
        410.95792223975747,
        354.04207776024253,
        354.04207776024253,
        455.95792223975747,
        399.04207776024253,
        399.04207776024253,
        500.95792223975747,
        444.04207776024253,
        444.04207776024253,
        545.9579222397574,
        489.04207776024253,
        489.04207776024253,
        590.9579222397574,
        534.0420777602426,
        534.0420777602426,
        635.9579222397574,
    ]

    for i, suffix in enumerate(suffixes):
        visualizeMask((math.radians(KEVXtheta[i]), KEVXradius[i] * 300))

        visualizeResults(
            "/Users/Kate/workspace/BirdRoostLocation/MLData/KEVX20130724_110326_V06/24KEVX20130724_110326_V06_Reflectivity"
            + suffix
            + ".png",
            (math.radians(KEVXtheta[i]), KEVXradius[i] * 300),
            (0, 0),
            "/Users/Kate/workspace/BirdRoostLocation/MLData/KEVX20130724_110326_V06/24KEVX20130724_110326_V06_Reflectivity"
            + suffix
            + ".png",
        )

    # for i in range(len(df)):
    #     aws_file = df["AWS_file"].iloc[[i]]
    #     theta = df["polar_theta"].iloc[[i]]
    #     radius = df["polar_radius"].iloc[[i]]

    #     for field in fields:
    #         full_path = (
    #             "/Users/Kate/workspace/BirdRoostLocation/MLData/highlights/Roost_"
    #             + field
    #             + "/"
    #             # + aws_file[i][10:12]
    #             + aws_file[i]
    #             + "_"
    #             + field
    #             + ".png"
    #         )
    #         save_path = (
    #             "/Users/Kate/workspace/BirdRoostLocation/MLData/all_true_data/Roost_"
    #             + field
    #             + "/"
    #             # + aws_file[i][10:12]
    #             + aws_file[i]
    #             + "_"
    #             + field
    #             + ".png"
    #         )
    #         try:
    #             print(settings.WORKING_DIRECTORY + "/" + settings.LABEL_CSV)
    #             print((math.radians(theta[i] + 180), radius[i]))
    #             print((theta[i], radius[i]))
    #             print(full_path)
    #             if os.path.isfile(full_path):
    #                 visualizeResults(
    #                     full_path,
    #                     # (math.radians(float(truth[1])), float(truth[0])),
    #                     (math.radians(theta[i] + 180), radius[i]),
    #                     (math.radians(theta[i]), radius[i]),
    #                     save_path,
    #                 )
    #         except:
    #             print(aws_file[i] + " was passed")
