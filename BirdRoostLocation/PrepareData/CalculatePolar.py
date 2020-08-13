import BirdRoostLocation.LoadSettings as settings
import os
import pandas
import csv
import math
import numpy as np


def calculate_polar(nexrad_lat, nexrad_long, roost_lat, roost_long):
    # define (nexrad_lat, nexrad_long) as (0, 0)
    # calculate difference and convert difference to polar
    roost_lat_km, roost_long_km = convert_km(roost_lat, roost_long)
    nexrad_lat_km, nexrad_long_km = convert_km(nexrad_lat, nexrad_long)

    lat_diff = roost_lat_km - nexrad_lat_km  # y
    long_diff = roost_long_km - nexrad_long_km  # x

    theta = math.degrees(math.atan(abs(lat_diff / long_diff)))
    if long_diff < 0 and lat_diff < 0:  # quadrant 3
        theta = theta - 180.0
    if long_diff < 0 and lat_diff >= 0:  # quadrant 2
        theta = theta + 180.0
    if lat_diff < 0 and long_diff >= 0:  # quadrant 4
        theta = -1 * theta

    rad = math.sqrt(lat_diff * lat_diff + long_diff * long_diff)

    return rad, theta  # lat/long degrees, degrees


def convert_km(latitude, longitude):
    latitude_km = latitude * 110.574
    longitude_km = longitude * 111.320 * math.cos(math.radians(latitude))
    return latitude_km, longitude_km


def create_nexrad_dict():
    with open(settings.NEXRAD_CSV, mode="r") as infile:
        reader = csv.reader(infile)
        nexrad_dict = {rows[0]: [rows[1], rows[2]] for rows in reader}
    return nexrad_dict


def add_cols():
    df = pandas.read_csv(
        "/Users/Kate/workspace/BirdRoostLocation/MLData/true_ml_relabels_edited.csv"
    )
    nexrads = create_nexrad_dict()

    nexrad_lats = []
    nexrad_longs = []
    polar_radii = []
    polar_theta = []
    for _, row in df.iterrows():
        nexrad_lat = nexrads[row["radar"]][0]
        nexrad_long = nexrads[row["radar"]][1]

        rad, theta = calculate_polar(
            float(nexrad_lat),
            float(nexrad_long),
            float(row["latitude"]),
            float(row["longitude"]),
        )

        if row["AWS_file"][0:19] == "KFSD20110820_113252":
            print(row["AWS_file"])
            print(float(row["latitude"]))
            print(float(row["longitude"]))
            print(rad)
            print(theta)

        polar_radii.append(rad)
        polar_theta.append(theta)
        nexrad_lats.append(nexrad_lat)
        nexrad_longs.append(nexrad_long)

    df["nexrad_lat"] = nexrad_lats
    df["nexrad_lon"] = nexrad_longs
    df["polar_radius"] = polar_radii
    df["polar_theta"] = polar_theta

    return df


def main():
    updated_df = add_cols()
    updated_df.to_csv(
        "/Users/Kate/workspace/BirdRoostLocation/MLData/true_ml_relabels_polar_edited.csv",
        index=False,
    )


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    main()
