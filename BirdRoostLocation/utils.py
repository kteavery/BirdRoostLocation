from enum import Enum
import os

RADAR_FILE_DIR = "radarfiles/"
# TODO change when running on schooner
RADAR_IMAGE_DIR = "/Users/Kate/workspace/BirdRoostLocation/MLData/"


class ML_Set(Enum):
    """Machine learning set enum, includes validation, train, and test."""

    def __new__(cls, value):
        enum_names = {0: "Validation", 1: "Training", 2: "Testing"}

        member = object.__new__(cls)
        member._value_ = value
        member.fullname = enum_names[value]
        return member

    def __int__(self):
        return self.value


class ML_Model(Enum):
    def __new__(cls, value):
        enum_names = {0: "Shallow_CNN", 1: "Shallow_CNN_All", 2: "Shallow_CNN_Time"}

        member = object.__new__(cls)
        member._value_ = value
        member.fullname = enum_names[value]
        return member

    def __int__(self):
        return self.value


class Radar_Products(Enum):
    """Radar Product enum, includes reflectivity, velocity, rho_hv, and zdr."""

    def __new__(cls, value):
        enum_names = {0: "Reflectivity", 1: "Velocity", 2: "Rho_HV", 3: "Zdr"}

        member = object.__new__(cls)
        member._value_ = value
        member.fullname = enum_names[value]
        return member

    def __int__(self):
        return self.value


Legacy_radar_products = [Radar_Products.reflectivity, Radar_Products.velocity]

pyart_key_dict = {
    Radar_Products.reflectivity: "reflectivity",
    Radar_Products.velocity: "velocity",
    Radar_Products.diff_reflectivity: "differential_reflectivity",
    Radar_Products.cc: "cross_correlation_ratio",
}


def getListOfFilesInDirectory(dir, fileType):
    """Given a folder, recursively return the names of all files of given type.

    Args:
        dir: path to folder, string
        fileType: Example: ".txt" or ".png"

    Returns:
        list of fileNames
    """
    fileNames = []
    for root, _, files in os.walk(dir):
        for f in files:
            if os.path.splitext(f)[1].lower() == fileType:
                fullPath = os.path.join(root, f)
                fileNames.append(fullPath)
    return fileNames
