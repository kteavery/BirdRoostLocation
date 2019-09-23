import BirdRoostLocation.LoadSettings as settings
import os
import pandas
import csv
import math


def calculate_polar(roost_lat, roost_long, nexrad_lat, nexrad_long):
    # define (nexrad_lat, nexrad_long) as (0, 0)
    # calculate difference and convert difference to polar
    lat_diff = roost_lat - nexrad_lat
    long_diff = roost_long - nexrad_long
    rad = math.sqrt(lat_diff * lat_diff + long_diff * long_diff)
    theta = math.degrees(math.atan(lat_diff / long_diff))
    return rad, theta  # lat/long degrees, radians


def create_nexrad_dict():
    with open(settings.NEXRAD_CSV, mode="r") as infile:
        reader = csv.reader(infile)
        nexrad_dict = {rows[0]: [rows[1], rows[2]] for rows in reader}
    return nexrad_dict


def add_cols():
    df = pandas.read_csv(settings.LABEL_CSV)
    nexrads = create_nexrad_dict()

    nexrad_lats = []
    nexrad_longs = []
    polar_radii = []
    polar_theta = []
    for row in df:
        nexrad_lat = nexrads[row["radar"]][0]
        nexrad_long = nexrads[row["radar"]][1]
        rad, theta = calculate_polar(row["lat"], row["long"], nexrad_lat, nexrad_long)

        nexrad_lats.append(nexrad_lat)
        nexrad_longs.append(nexrad_long)
        polar_radii.append(rad)
        polar_theta.append(theta)

    df["nexrad_lat"] = nexrad_lats
    df["nexrad_long"] = nexrad_longs
    df["polar_radius"] = polar_radii
    df["polar_theta"] = polar_theta


def main():
    add_cols()


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    main()
