import numpy as np
import os
import sys
import cv2
import scipy.misc
from PIL import Image

ROOT_DIR = "/datadrive/roost_data"

directory = ROOT_DIR
data_directories = [
    #"Roost_Reflectivity",
    #"Roost_Velocity",
    #"Roost_Zdr",
    #"Roost_Rho_HV",
    "NoRoost_Reflectivity",
    "NoRoost_Velocity",
    "NoRoost_Zdr",
    "NoRoost_Rho_HV",
]

flip_directories = ["Flip_" + i for i in data_directories]
data_directories.extend(flip_directories)
rotate_directories = ["Rotate_" + i for i in data_directories]
data_directories.extend(rotate_directories)

# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian
# -salt-and-pepper-etc-to-image-in-python-with-opencv


def salt_pepper(image):
    s_vs_p = 0.5
    amount = 0.02
    out = np.copy(image)
    # Salt
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255

    # Pepper
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out


def main():
    for data_dir in data_directories:
        for file in os.listdir(directory + "/data/" + data_dir):
            if file.endswith(".png") or file.endswith(".jpg"):
                image_ary = cv2.imread(directory + "/data/" + data_dir + "/" + file)

                noisy = salt_pepper(image_ary)

                filename = os.path.basename(file)
                if not os.path.exists(directory + "/data/Noise_" + data_dir):
                    os.makedirs(directory + "/data/Noise_" + data_dir)
                newfile = (
                    directory
                    + "/data/Noise_"
                    + data_dir
                    + "/"
                    + filename[:-4]
                    + "_noise.png"
                )
                cv2.imwrite(newfile, noisy)


if __name__ == "__main__":
    print("")
    main()
