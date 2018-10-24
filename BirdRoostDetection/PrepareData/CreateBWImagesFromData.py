import argparse
import os
import pandas

import BirdRoostDetection.LoadSettings as settings
from BirdRoostDetection.PrepareData import NexradUtils
from BirdRoostDetection import utils
from BirdRoostDetection.PrepareData import VisualizeNexradData


def plot_radar_files(file_names):
    for file_name in file_names:
        try:
            print file_name
            file_path = os.path.join(utils.RADAR_FILE_DIR,
                                     NexradUtils.getBasePath(file_name),
                                     file_name)
            img_path = file_path.replace(utils.RADAR_FILE_DIR,
                                         utils.RADAR_IMAGE_DIR + \
                                         '{0}/') + '_{0}.png'
            VisualizeNexradData.visualizeBWRadarData(file_path, img_path, True)
        except Exception as e:
            print '{}, {}'.format(file_name, str(e))


def main(results):
    """Formatted to run either locally or on schooner. Read in csv and get radar
     files listed in 'AWS_file' column. Save these files out as png images."""
    labels = pandas.read_csv(filepath_or_buffer=settings.LABEL_CSV,
                             skip_blank_lines=True)

    radar_labels = labels[labels.radar == results.radar]
    file_names = list(radar_labels['AWS_file'])
    plot_radar_files(file_names)


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
