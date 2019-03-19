import numpy as np
import os
import sys
import cv2
import scipy.misc
from PIL import Image
from scipy.ndimage import rotate

ROOT_DIR = "/Users/Kate/workspace/BirdRoostLocation"

directory = ROOT_DIR
data_directories = ["Roost_Reflectivity", "Roost_Velocity",
                    "Roost_Zdr", "Roost_Rho_HV", "NoRoost_Reflectivity",
                    "NoRoost_Velocity", "NoRoost_Zdr", "NoRoost_Rho_HV"]

flip_directories = ["Flip_"+i for i in data_directories]
data_directories.extend(flip_directories)


def main():
    for data_dir in data_directories:
        for file in os.listdir(directory+"/MLData/data/"+data_dir):
            if file.endswith(".png") or file.endswith(".jpg"):
                image_ary = cv2.imread(directory+"/MLData/data/" +
                                       data_dir+"/"+file)

                for angle in np.arange(45, 360, 45):
                    rotated = scipy.ndimage.rotate(image_ary, angle, cval=255,
                                                   reshape=False)

                    filename = os.path.basename(file)
                    os.makedirs(directory + "/MLData/data/Rotate_" + data_dir,
                                exist_ok=True)
                    newfile = directory + "/MLData/data/Rotate_" + data_dir + \
                        "/" + filename[:-4] + "_" + str(angle) + ".png"
                    cv2.imwrite(newfile, rotated)


if __name__ == "__main__":
    main()
