from enum import Enum
import os
from typing import Dict, Tuple, List

RADAR_FILE_DIR: str = "radarfiles/"
# TODO change when running on schooner
RADAR_IMAGE_DIR: str = "/datadrive/roost_data/"


class ML_Set(Enum):
    """Machine learning set enum, includes validation, train, and test."""

    validation: Tuple[int, str] = (0, "Validation")
    training: Tuple[int, str] = (1, "Training")
    testing: Tuple[int, str] = (2, "Testing")

    def __new__(cls, value: int, name: str):
        # enum_names = {0: "Validation", 1: "Training", 2: "Testing"}
        member: Tuple[int, str] = object.__new__(cls)
        member._value_: int = value
        member.fullname: str = name  # enum_names[value]
        return member

    def __int__(self) -> int:
        return self.value


class ML_Model(Enum):
    Shallow_CNN: Tuple[int, str] = (0, "Shallow_CNN")
    Shallow_CNN_All: Tuple[int, str] = (1, "Shallow_CNN_All")
    Shallow_CNN_Time: Tuple[int, str] = (2, "Shallow_CNN_Time")

    def __new__(cls, value: int, name: str):
        # enum_names = {0: "Shallow_CNN", 1: "Shallow_CNN_All", 2: "Shallow_CNN_Time"}
        member: Tuple[int, str] = object.__new__(cls)
        member._value_: int = value
        member.fullname: str = name  # enum_names[value]
        return member

    def __int__(self) -> int:
        return self.value


class Radar_Products(Enum):
    """Radar Product enum, includes reflectivity, velocity, rho_hv, and zdr."""

    reflectivity: Tuple[int, str] = (0, "Reflectivity")
    velocity: Tuple[int, str] = (1, "Velocity")
    cc: Tuple[int, str] = (2, "Rho_HV")
    diff_reflectivity: Tuple[int, str] = (3, "Zdr")

    def __new__(cls, value: int, name: str) -> Tuple[int, str]:
        # enum_names = {0: "Reflectivity", 1: "Velocity", 2: "Rho_HV", 3: "Zdr"}
        member: Tuple[int, str] = object.__new__(cls)
        member._value_: int = value
        member.fullname: str = name  # enum_names[value]
        return member

    def __int__(self) -> int:
        return self.value


Legacy_radar_products: List[Tuple[int, str]] = [
    Radar_Products.reflectivity,
    Radar_Products.velocity,
]

pyart_key_dict: Dict[tuple, str] = {
    Radar_Products.reflectivity: "reflectivity",
    Radar_Products.velocity: "velocity",
    Radar_Products.diff_reflectivity: "differential_reflectivity",
    Radar_Products.cc: "cross_correlation_ratio",
}


def getListOfFilesInDirectory(dir: str, fileType: str) -> List[str]:
    """Given a folder, recursively return the names of all files of given type.

    Args:
        dir: path to folder, string
        fileType: Example: ".txt" or ".png"

    Returns:
        list of fileNames
    """
    fileNames: List[str] = []
    for root, _, files in os.walk(dir):
        for f in files:
            if os.path.splitext(f)[1].lower() == fileType:
                fullPath = os.path.join(root, f)
                fileNames.append(fullPath)
    return fileNames
