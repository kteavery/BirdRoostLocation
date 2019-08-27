import os
import datetime
import numpy as np
from PIL import Image
from BirdRoostLocation import utils
import ast
import re


class ML_Label:
    """This class contains all the information needed for a single ML label.

    Class Variables:
        self.fileName: The filename, RRRRYYYYMMDD_HHMMSS_VO#
        self.is_roost: True is it the file contains a roost
        self.roost_id: The id of the roost, -1 if no roost
        self.latitude: The latitude of the roost in the radar file
        self.longitude: The longitude of the roost in the radar file
        self.timestamp: The radius of the roost in the radar file
        self.sunrise_time: The sunrise time at the lat, lon coordinates
        self.image_paths: A dictionary that contains path to the images with the
            following radar products : reflectivity, velocity, rho_hv, zdr
    """

    def __init__(self, file_name, pd_row, root_dir, high_memory_mode):
        """Initialize class using roost directory and pandas dataframe row.

        Args:
            pd_row: A pandas dataframe row read in from a csv with the format
            in ml_labels_example.csv
            root_dir: The directory where the images as stored.
            high_memory_mode: Boolean, if true then all of the data will be read
            in at the beginning and stored in memeory. Otherwise only one batch
            of data will be in memeory at a time. high_memory_mode is good
            for machines with slow IO and at least 8 GB of memeory available.

        """
        self.high_memory_mode = high_memory_mode
        self.fileName = file_name
        self.is_roost = pd_row["Roost"]
        self.roost_id = pd_row["roost_id"]
        self.latitude = pd_row["lat"]
        self.longitude = pd_row["lon"]
        self.radius = pd_row["radius"]
        self.timestamp = datetime.datetime.strptime(
            pd_row["roost_time"], "%Y-%m-%d %H:%M:%S"
        )
        self.sunrise_time = datetime.datetime.strptime(
            pd_row["sunrise_time"], "%Y-%m-%d %H:%M:%S"
        )
        self.images = {}
        for radar_product in utils.Radar_Products:
            image_path = self.__get_augmented_product_paths(
                root_dir, radar_product.fullname, self.is_roost
            )
            if self.high_memory_mode:
                paths = []
                for path in image_path:
                    paths.append(self.load_image(path))
                self.images[radar_product] = paths
            else:
                self.images[radar_product] = image_path

    def __str__(self):
        return (
            str(self.fileName)
            + ", "
            + str(self.is_roost)
            + ", "
            + str(self.roost_id)
            + ", "
            + str(self.latitude)
            + ", "
            + str(self.longitude)
            + ", "
            + str(self.radius)
            + ", "
            + str(self.timestamp)
            + ", "
            + str(self.sunrise_time)
            + ", "
            + str(self.high_memory_mode)
            + ", "
            + str(self.images)
        )

    def get_image(self, radar_product):
        if self.high_memory_mode:
            return self.images[radar_product]
        if isinstance(self.images[radar_product], (list,)):
            images = []
            for image in self.images[radar_product]:
                if self.load_image(image) is not None:
                    images.append(self.load_image(image))
            return images
        else:
            return self.load_image(self.images[radar_product])

    def __get_radar_product_path(self, root_dir, radar_product, is_roost):
        if is_roost:
            return os.path.join(
                root_dir, "data/Roost_" + "{1}/", "{2}{0}_{1}" + ".png"
            ).format(self.fileName, radar_product, self.fileName[10:12])
        else:
            return os.path.join(
                root_dir, "data/NoRoost_" + "{1}/", "{2}{0}_{1}" + ".png"
            ).format(self.fileName, radar_product, self.fileName[10:12])

    def __get_augmented_product_paths(self, root_dir, radar_product, is_roost):
        paths = []
        for roost in ["Roost_", "NoRoost_"]:
            paths.extend(
                [
                    os.path.join(
                        root_dir + "data/" + roost + "{1}/" + "{2}{0}_{1}.png"
                    ).format(self.fileName, radar_product, self.fileName[10:12]),
                    os.path.join(
                        root_dir + "data/Flip_" + roost + "{1}/" + "{2}{0}_{1}_flip.png"
                    ).format(self.fileName, radar_product, self.fileName[10:12]),
                    os.path.join(
                        root_dir
                        + "data/Noise_Flip_"
                        + roost
                        + "{1}/"
                        + "{2}{0}_{1}_flip_noise.png"
                    ).format(self.fileName, radar_product, self.fileName[10:12]),
                    os.path.join(
                        root_dir, "data/Noise_" + roost + "{1}/", "{2}{0}_{1}_noise.png"
                    ).format(self.fileName, radar_product, self.fileName[10:12]),
                ]
            )

            for angle in ["45", "90", "135", "180", "225", "270", "315"]:
                paths.extend(
                    [
                        os.path.join(
                            root_dir,
                            "data/Noise_Rotate_" + roost + "{1}/",
                            "{2}{0}_{1}_" + angle + "_noise.png",
                        ).format(self.fileName, radar_product, self.fileName[10:12]),
                        os.path.join(
                            root_dir,
                            "data/Rotate_" + roost + "{1}/",
                            "{2}{0}_{1}_" + angle + ".png",
                        ).format(self.fileName, radar_product, self.fileName[10:12]),
                        os.path.join(
                            root_dir,
                            "data/Rotate_Flip_" + roost + "{1}/",
                            "{2}{0}_{1}_flip_" + angle + ".png",
                        ).format(self.fileName, radar_product, self.fileName[10:12]),
                    ]
                )
        return paths

    def load_image(self, filename):
        """Load image from filepath.

        Args:
            filename: The path to the image file.

        Returns:
            Image as numpy array.
        """
        dim = 120
        if not os.path.exists(filename):
            # print(filename)
            return None
        img = Image.open(filename)
        img_rgb = np.array(img.convert("RGB"))
        shape = img_rgb.shape
        w_mid = int(shape[0] / 2)
        h_mid = int(shape[1] / 2)
        img_rgb = img_rgb[w_mid - dim : w_mid + dim, h_mid - dim : h_mid + dim]
        return img_rgb


class Temporal_ML_Label(ML_Label):
    def __init__(self, file_name, pd_row, root_dir, high_memory_mode, label_dict):
        if not (file_name in label_dict):
            ML_Label.__init__(self, file_name, pd_row, root_dir, high_memory_mode)

            # ast.literal_eval(pd_row['AWS_file'])
            self.fileNames = pd_row["AWS_file"]

            label_dict[file_name] = self
            for name in self.fileNames:
                Temporal_ML_Label(name, pd_row, root_dir, high_memory_mode, label_dict)

    def __str__(self):
        return ML_Label.__str__(self)


class Color_ML_Label(ML_Label):
    def __init__(self, file_name, pd_row, root_dir, high_memory_mode):
        ML_Label.__init__(self, file_name, pd_row, root_dir, high_memory_mode)
        for radar_product in utils.Radar_Products:
            image_path = self.__get_radar_product_path(root_dir, radar_product.fullname)
            self.images[radar_product] = image_path

    def __get_radar_product_path(self, root_dir, radar_product):
        return os.path.join(root_dir, "{1}_Color/", "{0}_{1}.png").format(
            self.fileName, radar_product
        )

    def __str__(self):
        return ML_Label.__str__(self)
