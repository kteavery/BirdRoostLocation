"""
Once all NEXRAD radar images have been downloaded, convert them to images.

In order to parallelize the process of making images of the data, we run files
from each radar separately. For our research we had 81 radars and ran this file
81 times in parallel on schooner (OU super computer)

Example command:
python CreateImagesFromData.py KLIX
"""
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.PrepareData import VisualizeNexradData
from BirdRoostLocation.PrepareData import NexradUtils
from BirdRoostLocation import utils
import os
import argparse
import pyart.io
from PIL import Image
import pandas
import sys
import glob


def createLabelForFiles(fileNames, saveDir, radarFilePath):
    """Given a Level 2 NEXRAD radar file, create images.

    This is a slightly fast and lazy was of creating these images. There is
    probably a better way to do this but this functions for my purposes
    I use the PyArt library to save out the radar products, and then read in the
    images and save out the 4 individual radar products.

    Args:
        fileNames: A list of filename paths, the location of the NEXRAD radar
            files.
        saveDir: The directory where the images will be saved in.
    """
    labelDF = pandas.read_csv(settings.WORKING_DIRECTORY + "/" + settings.LABEL_CSV)

    for f in fileNames:

        root = os.path.join(radarFilePath, NexradUtils.getBasePath(f))
        name = f.replace(".gz", "")

        # print("SAVEDIR: ")
        # print(saveDir)
        imgDir = os.path.join(saveDir, NexradUtils.getBasePath(f)) + "/"
        imgPath = os.path.join(
            imgDir.replace(saveDir, os.path.join(saveDir, "All_Color")), name + ".png"
        )
        print("IMGPATH:")
        print(imgPath)

        file = open(os.path.join(root, name), "r")
        if not os.path.exists(os.path.dirname(imgPath)):
            os.makedirs(os.path.dirname(imgPath))

        rad = pyart.io.read_nexrad_archive(file.name)

        dualPol = int(name[-1:]) >= 6

        label_row = labelDF.loc[labelDF["AWS_file"] == f]
        for i in range(len(label_row)):
            lat = float(label_row["lat"].iloc[[i]])
            lon = float(label_row["lon"].iloc[[i]])
            # print(float(label_row["nexrad_lat"].iloc[[i]]))
            # print(float(label_row["nexrad_lon"].iloc[[i]]))
            VisualizeNexradData.visualizeRadarData(
                rad,
                imgPath[:-4] + ".png",
                dualPol,
                nexrads=[
                    float(label_row["nexrad_lat"].iloc[[i]]),
                    float(label_row["nexrad_lon"].iloc[[i]]),
                ],
            )

        file.close()

        saveAndSplitImages(imgDir, saveDir, dualPol, imgPath, name)


def createWithoutCSV(fileNames, saveDir, radarFilePath):
    for k, f in enumerate(fileNames):
        # root = os.path.join(radarFilePath, NexradUtils.getBasePath(f))
        name = f.replace(".gz", "")

        # print("SAVEDIR: ")
        # print(saveDir)
        imgPath = os.path.join(saveDir, "2019images", name + ".png")
        # print("IMGPATH:")
        # print(imgPath)

        file = open(os.path.join(radarFilePath, name + ".gz"), "r")
        if not os.path.exists(os.path.dirname(imgPath)):
            os.makedirs(os.path.dirname(imgPath))

        rad = pyart.io.read_nexrad_archive(file.name)

        dualPol = int(name[-1:]) >= 6

        nexrad_csv = pandas.read_csv(
            saveDir + "/nexrad.csv", names=["radar", "lat", "lon"]
        )

        nexrad_row = nexrad_csv[nexrad_csv["radar"] == fileNames[k][0:4]]
        VisualizeNexradData.visualizeRadarData(
            rad,
            imgPath[:-4] + ".png",
            dualPol,
            nexrads=[float(nexrad_row["lat"]), float(nexrad_row["lon"])],
        )

        saveAndSplitImages(saveDir + "2019images/", saveDir, dualPol, imgPath, name)


def saveAndSplitImages(imgDir, saveDir, dualPol, imgPath, name):
    d1 = imgDir.replace(saveDir, os.path.join(saveDir, "Reflectivity_Color"))
    d2 = imgDir.replace(saveDir, os.path.join(saveDir, "Velocity_Color"))
    if dualPol:
        d3 = imgDir.replace(saveDir, os.path.join(saveDir, "Rho_HV_Color"))
        d4 = imgDir.replace(saveDir, os.path.join(saveDir, "Zdr_Color"))

    if not os.path.exists(d1):
        os.makedirs(d1)
    if not os.path.exists(d2):
        os.makedirs(d2)
    if dualPol:
        if not os.path.exists(d3):
            os.makedirs(d3)
        if not os.path.exists(d4):
            os.makedirs(d4)

    img = Image.open(imgPath)

    save_extension = ".png"
    if not dualPol:
        img1 = img.crop((115, 100, 365, 350))
        # print("DIR NAMES: ")
        # print(d1 + name + "_Reflectivity" + save_extension)
        img1.save(d1 + name + "_Reflectivity" + save_extension)

        img2 = img.crop((495, 100, 740, 350))
        # print(d2 + name + "_Velocity" + save_extension)
        img2.save(d2 + name + "_Velocity" + save_extension)

    if dualPol:
        img1 = img.crop((115, 140, 365, 390))
        # print("DIR NAMES: ")
        # print(d1 + name + "_Reflectivity" + save_extension)
        img1.save(d1 + name + "_Reflectivity" + save_extension)

        img2 = img.crop((495, 140, 740, 390))
        # print(d2 + name + "_Velocity" + save_extension)
        img2.save(d2 + name + "_Velocity" + save_extension)

        img3 = img.crop((115, 520, 365, 770))
        # print(d3 + name + "_Zdr" + save_extension)
        img3.save(d3 + name + "_Zdr" + save_extension)

        img4 = img.crop((495, 520, 740, 770))
        # print(d4 + name + "_Rho_HV" + save_extension)
        img4.save(d4 + name + "_Rho_HV" + save_extension)


def main(results):
    """Formatted to run either locally or on schooner. Read in csv and get radar
     files listed in 'AWS_file' column. Save these files out as png images."""
    labels = pandas.read_csv(
        filepath_or_buffer=settings.LABEL_CSV, skip_blank_lines=True
    )

    radar_labels = labels[labels.radar == results.radar]
    createLabelForFiles(
        fileNames=list(radar_labels["AWS_file"]),
        saveDir=utils.RADAR_IMAGE_DIR,
        radarFilePath="radarfiles/",
    )
    aws_files = []
    for file in glob.glob(utils.RADAR_IMAGE_DIR + "/2019radarfiles/" + "*.gz"):
        if file[-6:-3] != "MDM" and not os.path.isfile(
            utils.RADAR_IMAGE_DIR + "/2019images/" + file[-26:-3] + ".png"
        ):
            aws_files.append(os.path.basename(file)[0:23])
        # else:
        #     print(file)
    # print(aws_files)
    print(len(aws_files))

    # createWithoutCSV(
    #     fileNames=aws_files,
    #     saveDir=utils.RADAR_IMAGE_DIR,
    #     radarFilePath="2019radarfiles/",
    # )
    try:
        createWithoutCSV(
            fileNames=aws_files,
            saveDir=utils.RADAR_IMAGE_DIR,
            radarFilePath="2019radarfiles/",
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--radar",
        type=str,
        default="KIND",
        help="""A 4 letter key of a USA NEXRAD radar. Example: KLIX""",
    )
    results = parser.parse_args()
    main(results)
