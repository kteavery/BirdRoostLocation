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

    print(roost_lat_km)
    print(nexrad_lat_km)
    print(roost_long_km)
    print(nexrad_long_km)
    lat_diff = roost_lat_km - nexrad_lat_km  # y
    long_diff = roost_long_km - nexrad_long_km  # x
    print(lat_diff)
    print(long_diff)

    theta = math.degrees(math.atan(abs(lat_diff / long_diff)))
    if long_diff < 0 and lat_diff < 0:  # quadrant 3
        theta = -1 * (180.0 - theta)
    if long_diff < 0 and lat_diff >= 0:  # quadrant 2
        theta = 180.0 - theta
    if lat_diff < 0 and long_diff >= 0:  # quadrant 4
        theta = -1 * theta
    if long_diff >= 0 and lat_diff >= 0:
        print("quadrant 1")

    print(theta)

    rad = math.sqrt(lat_diff * lat_diff + long_diff * long_diff)

    print(rad)
    print()

    return rad, theta  # lat/long degrees, degrees


def convert_km(latitude, longitude):
    latitude_km = latitude * 111.11
    longitude_km = longitude * 111.0 * math.cos(math.radians(latitude))
    return latitude_km, longitude_km


def create_nexrad_dict():
    with open(settings.NEXRAD_CSV, mode="r") as infile:
        reader = csv.reader(infile)
        nexrad_dict = {rows[0]: [rows[1], rows[2]] for rows in reader}
    return nexrad_dict


def add_cols():
    df = pandas.read_csv(settings.LABEL_CSV)
    nexrads = create_nexrad_dict()
    # print(df)

    nexrad_lats = []
    nexrad_longs = []
    polar_radii = []
    polar_theta = []
    for _, row in df.iterrows():
        nexrad_lat = nexrads[row["radar"]][0]
        nexrad_long = nexrads[row["radar"]][1]

        rad, theta = calculate_polar(
            float(nexrad_lat), float(nexrad_long), float(row["lat"]), float(row["lon"])
        )

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
    print(calculate_polar(37.0242098, -80.2736664, 36.113, -80.61))
    print(calculate_polar(33.9487579, -81.1184281, 33.37, -80.0))
    print(calculate_polar(38.9753957, -77.4778444, 38.3667, -76.9177))
    # updated_df = add_cols()
    # updated_df.to_csv(settings.UPDATE_LABEL_CSV, index=False)


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    main()
