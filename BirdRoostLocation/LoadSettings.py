"""Load in variables that are user specific and consistently references.

This repository's scrips will read and save files to a given directory. Set a
working directory where all these files will be saved. BirdRoostDetection/MLData
is the default however any location can be chosen. All radar files, image files,
model files, and data csv files will be saved in this directory. Mose of the
scrips in this repository will will assume this folder is the working directory
and all paths can be relative to this directory.

Set file paths in the settings.json file.
"""

import json
import os
import matplotlib
from typing import Dict, Any

real_path: str = os.path.realpath(__file__)
setting_path: str = os.path.join(os.path.dirname(real_path), "settings.json")
data: Dict[str, Any] = json.load(open(setting_path))

WORKING_DIRECTORY: str = str(data["cwd"])
LABEL_CSV: str = str(data["label_csv"])
PREDICTION_CSV: str = str(data["prediction_csv"])
UPDATE_LABEL_CSV: str = str(data["update_label_csv"])
SUBSET_CSV: str = str(data["subset_files_csv"])
ML_SPLITS_DATA: str = str(data["ml_splits_csv"])
NEXRAD_CSV: str = str(data["nexrad_csv"])
DEFAULT_BATCH_SIZE: int = 1

if bool(data["schooner"]):
    print("schooner")
    matplotlib.use("agg")  # Required for running on schooner
