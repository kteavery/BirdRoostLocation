import numpy as np
import os
import sys
import cv2
import scipy.misc
from PIL import Image

ROOT_DIR = "/Users/Kate/workspace/BirdRoostLocation"
#ROOT_DIR = "/condo/swatwork/keavery/masters_thesis"

directory = ROOT_DIR
data_directories = ["Roost_Reflectivity", "Roost_Velocity",
                    "Roost_Zdr", "Roost_Rho_HV", "NoRoost_Reflectivity",
                    "NoRoost_Velocity", "NoRoost_Zdr", "NoRoost_Rho_HV"]


def main():
    for data_dir in data_directories:
        for file in os.listdir(directory+"/MLData/data/"+data_dir):
            if file.endswith(".png") or file.endswith(".jpg"):
                image_ary = cv2.imread(directory+"/MLData/data/" +
                                       data_dir+"/"+file)

                flipped = np.fliplr(image_ary)
                filename = os.path.basename(file)
                os.makedirs(directory + "/MLData/data/Flip_" + data_dir,
                            exist_ok=True)
                newfile = directory + "/MLData/data/Flip_" + data_dir + "/" \
                    + filename[:-4] + "_flip.png"

                cv2.imwrite(newfile, flipped)


if __name__ == "__main__":
    print("")
    main()
