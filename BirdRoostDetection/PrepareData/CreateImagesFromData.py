"""Once all NEXRAD radar images have been downloaded, convert them to images.

In order to parallelize the process of making images of the data, we run files
from each radar separately. For our research we had 81 radars and ran this file
81 times in parallel on schooner (OU super computer)

Example command:
python CreateImagesFromData.py KLIX
"""
import BirdRoostDetection.LoadSettings as settings
from BirdRoostDetection.PrepareData import VisualizeNexradData
from BirdRoostDetection.PrepareData import NexradUtils
from BirdRoostDetection import utils
import os
import argparse
import pyart.io
from PIL import Image
import pandas


def createLabelForFiles(fileNames, saveDir):
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
    radarFilePath = 'radarfiles/'
    for f in fileNames:
        try:
            root = os.path.join(radarFilePath, NexradUtils.getBasePath(f))
            name = f.replace('.gz', '')
            imgDir = os.path.join(saveDir, NexradUtils.getBasePath(f)) + '/'
            imgPath = os.path.join(
                imgDir.replace(saveDir, os.path.join(saveDir, 'All_Color/')),
                name + '.png')
            print imgPath

            if not os.path.isfile(imgPath):
                file = open(os.path.join(root, name), 'r')
                if not os.path.exists(os.path.dirname(imgPath)):
                    os.makedirs(os.path.dirname(imgPath))

                rad = pyart.io.read_nexrad_archive(file.name)

                dualPol = int(name[-1:]) >= 6
                VisualizeNexradData.visualizeRadardata(rad, imgPath, dualPol)
                file.close()

                d1 = imgDir.replace(saveDir, os.path.join(saveDir,
                                                          'Reflectivity_Color/'))
                d2 = imgDir.replace(saveDir, os.path.join(saveDir,
                                                          'Velocity_Color/'))
                if dualPol:
                    d3 = imgDir.replace(saveDir, os.path.join(saveDir,
                                                              'Differential_Reflectivity_Color/'))
                    d4 = imgDir.replace(saveDir, os.path.join(saveDir,
                                                              'Differential_Reflectivity_Color/'))

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
                save_extension = '.png'
                if (not dualPol):
                    img1 = img.crop((115, 100, 365, 350))
                    img1.save(d1 + name + '_Reflectivity' + save_extension)

                    img2 = img.crop((495, 100, 740, 350))
                    img2.save(d2 + name + '_Velocity' + save_extension)

                if (dualPol):
                    img1 = img.crop((115, 140, 365, 390))
                    img1.save(d1 + name + '_Reflectivity' + save_extension)

                    img2 = img.crop((495, 140, 740, 390))
                    img2.save(d2 + name + '_Velocity' + save_extension)

                    img3 = img.crop((115, 520, 365, 770))
                    img3.save(d3 + name + '_Zdr' + save_extension)

                    img4 = img.crop((495, 520, 740, 770))
                    img4.save(d4 + name + '_Rho_HV' + save_extension)

                    # print root + '/' + name
        except Exception as e:
            print '{}, {}'.format(imgPath, str(e))


def main(results):
    """Formatted to run either locally or on schooner. Read in csv and get radar
     files listed in 'AWS_file' column. Save these files out as png images."""
    labels = pandas.read_csv(filepath_or_buffer=settings.LABEL_CSV,
                             skip_blank_lines=True)

    radar_labels = labels[labels.radar == results.radar]
    createLabelForFiles(fileNames=list(radar_labels['AWS_file']),
                        saveDir=utils.RADAR_IMAGE_DIR)


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--radar',
        type=str,
        default='KLIX',
        help=""" A 4 letter key of a USA NEXRAD radar. Example: KLIX"""
    )
    results = parser.parse_args()
    main(results)
