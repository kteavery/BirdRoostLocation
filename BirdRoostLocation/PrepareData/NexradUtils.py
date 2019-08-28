import pyart.io.nexrad_common as nexrad
import sys
from gpxpy import geo
import os
import datetime


def getRadarNames(minLat, maxLat, minLon, maxLon):
    """Given bounding lat/lon, this method returns all radar names in area.

    Args:
        minLat: Minimum Latitude, float.
        maxLat: Maximum Latitude, float.
        minLon: Minimum Longitude, float.
        maxLon: Maximum Longitude, float.

    Returns:
        List of radar names.
    """
    radars = []
    for key in list(nexrad.NEXRAD_LOCATIONS.keys()):
        if key[0] == "K":
            lat = float(nexrad.NEXRAD_LOCATIONS[key]["lat"])
            lon = float(nexrad.NEXRAD_LOCATIONS[key]["lon"])
            if lat >= minLat and lat <= maxLat and lon >= minLon and lon <= maxLon:
                radars.append(key)
    return radars


def getRadarLocation(radar):
    """Get the location of a given radar.

    Args:
        radar: 4 character name of radar, string.

    Returns:
        A list: [lat, lon]
    """
    return [
        nexrad.NEXRAD_LOCATIONS[radar]["lat"],
        nexrad.NEXRAD_LOCATIONS[radar]["lon"],
    ]


def getRadarsInRadius(lat, lon, radius_km=300.0):
    """Get all radars within a specified radius centered at lat, lon.

    Args:
        lat: Latitude, float.
        lon: Longitude, float.
        radius_km: The radius to search for radars in kilometers, float.
            Default is 300.0 km.

    Returns:
        A list of 4 character radar name strings.
    """
    radars = []
    for key in list(nexrad.NEXRAD_LOCATIONS.keys()):
        distance_km = getDistanceFromRadar(lat, lon, key)
        if distance_km < radius_km:
            radars.append(key)
    return radars


def getDistanceFromRadar(lat, lon, radar):
    """Find the distance between a lat, lon coordinate and a given radar.

    Args:
        lat: Latitude, float.
        lon: Longitude, float.
        radar: 4 character name of radar, string.

    Returns:
        distance in kilometers, float
    """
    radar_lat = float(nexrad.NEXRAD_LOCATIONS[radar]["lat"])
    radar_lon = float(nexrad.NEXRAD_LOCATIONS[radar]["lon"])
    distance_km = geo.haversine_distance(lat, lon, radar_lat, radar_lon) / 1000
    return distance_km


def getClosestRadar(lat, lon):
    """Given a lat, lon coordinate, find the closest radar.

    Args:
        lat: Latitude, float.
        lon: Longitude, float.

    Returns:
        (radar, distance) : The radar a a 4 letter string and distance is a
        float, the distance in kilometers.
    """
    radar = None
    min_distance = sys.maxsize
    for key in list(nexrad.NEXRAD_LOCATIONS.keys()):
        distance = getDistanceFromRadar(lat, lon, key)
        if min_distance > distance:
            radar = key
            min_distance = distance
    return radar, min_distance


def getTimeStampFromFilename(filename):
    """Get the timestamp from the AWS filename.

    Args:
        filename: The name of the AWS file.

    Returns:
        datetime object, the date corresponding with the radar datum timestamp.
    """
    base_f = os.path.basename(filename)
    radar_date = datetime.datetime.strptime(base_f[4:19], "%Y%m%d_%H%M%S")
    return radar_date


def getRadarFromFilename(filename):
    """Get the radar name from the AWS filename.

    Args:
        filename: The name of the AWS file.

    Returns:
        4 character name of radar, string.
    """
    base_f = os.path.basename(filename)
    radar = base_f[0:4]
    return radar


def getBasePath(radarFileName):
    """Given a single Nexrad radar file, create a path to save file at.

    In order to avoid saving too many files in a single folder, we save radar
    files and image in a path order using radar/year/month/day.

    Args:
        radarFileName: The name of the NEXRAD radar file.

    Returns:
        string path, RRRR/YYYY/MM/DD
    """
    radarFileName = os.path.basename(radarFileName)
    return os.path.join(
        radarFileName[0:4],
        radarFileName[4:8],
        radarFileName[8:10],
        radarFileName[10:12],
    )

