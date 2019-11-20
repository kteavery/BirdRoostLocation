import matplotlib.pyplot as plt
import math
import pandas
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
    # create axes in the background to show cartesian image
    ax0 = fig.add_subplot(111)
    # ax0.axis("off")
    ax0.imshow(data)

    # create polar axes in the foreground and remove its background
    # to see through
    ax = fig.add_subplot(111, polar=True, label="polar")
    ax.set_facecolor("None")
    ax.scatter(truth[0], truth[1], c="black", marker="*", s=70.0)
    # ax.scatter(prediction[0], prediction[1], c="black", marker="o")
    ax.axis("off")
    ax.set_rmax(300)

    plt.show()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    df = pandas.read_csv(settings.WORKING_DIRECTORY + "/" + settings.LABEL_CSV)
    fields = ["Reflectivity", "Velocity", "Rho_HV", "Zdr"]

    for i in range(len(df)):
        aws_file = df["AWS_file"].iloc([i])
        theta = df["polar_theta"].iloc([i])
        radius = df["polar_radius"].iloc([i])
        # truth = df["truth"].iloc([i])
        # prediction = df["prediction"].iloc([i])

        # truth = re.findall(r"[-+]?\d*\.\d+|\d+", truth[i])
        # prediction = re.findall(r"[-+]?\d*\.\d+|\d+", prediction[i])

        print(aws_file[i])
        print(theta[i])
        print(radius[i])

        for field in fields:
            full_path = (
                "/Users/Kate/workspace/BirdRoostLocation/MLData/all_true_data/Roost_"
                + field
                + "/"
                + aws_file[i][10:12]
                + aws_file[i]
                + "_"
                + field
                + ".png"
            )
            save_path = (
                "/Users/Kate/workspace/BirdRoostLocation/MLData/predictions/Roost_"
                + field
                + "/"
                + aws_file[i][10:12]
                + aws_file[i]
                + "_"
                + field
                + ".png"
            )

            print("ISFILE")
            print(os.path.isfile(full_path))
            try:
                if os.path.isfile(full_path):
                    visualizeResults(
                        full_path,
                        # (math.radians(float(truth[1])), float(truth[0])),
                        (math.radians(theta[i]), radius[i]),
                        # (float(prediction[1]), float(prediction[0]) * 300),
                        (0, 0),
                        save_path,
                    )
            except:
                print(aws_file[i] + " was passed")
