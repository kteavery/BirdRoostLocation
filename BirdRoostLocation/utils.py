from enum import Enum
import os

RADAR_FILE_DIR = 'radarfiles/'
RADAR_IMAGE_DIR = 'radarimages/'


class ML_Set(Enum):
    """Machine learning set enum, includes validation, train, and test."""
    validation = 0, 'Validation'
    training = 1, 'Training'
    testing = 2, 'Testing'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class ML_Model(Enum):
    Shallow_CNN = 0, 'Shallow_CNN'
    Shallow_CNN_All = 1, 'Shallow_CNN_All'
    Shallow_CNN_Time = 2, 'Shallow_CNN_Time'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class Radar_Products(Enum):
    """Radar Product enum, includes reflectivity, velocity, rho_hv, and zdr."""
    reflectivity = 0, 'Reflectivity'
    velocity = 1, 'Velocity'
    cc = 2, 'Correlation_Coefficient'
    diff_reflectivity = 3, 'Differential_Reflectivity'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


Legacy_radar_products = [Radar_Products.reflectivity, Radar_Products.velocity]

pyart_key_dict = {
    Radar_Products.reflectivity: 'reflectivity',
    Radar_Products.velocity: 'velocity',
    Radar_Products.diff_reflectivity: 'differential_reflectivity',
    Radar_Products.cc: 'cross_correlation_ratio'
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
    for root, dirs, files in os.walk(dir):
        for f in files:
            if os.path.splitext(f)[1].lower() == fileType:
                fullPath = os.path.join(root, f)
                fileNames.append(fullPath)
    return fileNames
