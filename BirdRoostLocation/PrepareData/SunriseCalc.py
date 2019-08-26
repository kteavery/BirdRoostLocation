"""Calculates sunrise for a given day in UTC.

The code in Sunrise class was modified from jebeaudet at https://github.com/
# jebeaudet/SunriseSunsetCalculator/blob/master/sunrise_sunset.py
"""

import math
import datetime


def calculate_sunrise(year, month, day, latitude, longitude):
    """ Get sunrise time in UTC given year, month, day, lat, and lon.

    Args:
        year: The year, integer (e.g. 2018)
        month: The month, integer (e.g. 2)
        day: The day, integer (e.g. 10)
        latitude: The latitude, float. North=positive, South=negative (e.g.
            35.222569)
        longitude: The latitude, float. East=positive, West=negative (e.g.
            -97.439476)

    Returns:
        Datetime object, the sunrise time in UTC
    """
    if latitude < -90 or latitude > 90:
        raise ValueError("Invalid latitude value")
    if longitude < -180 or longitude > 180:
        raise ValueError("Invalid longitude value")

    zenith = 90.83333

    # date
    dt = datetime.date(year, month, day)

    # weekday
    dow = dt.weekday()

    # Calculate the day of the year
    N = dt.toordinal() - datetime.date(dt.year, 1, 1).toordinal() + 1

    # Convert the longitude to hour value and calculate an approximate time
    lngHour = longitude / 15
    t_rise = N + ((6 - lngHour) / 24)
    t_set = N + ((18 - lngHour) / 24)

    # Calculate the Sun's mean anomaly
    M_rise = (0.9856 * t_rise) - 3.289

    # Calculate the Sun's true longitude, and adjust angle to be between 0
    # and 360
    L_rise = (
        M_rise
        + (1.916 * math.sin(math.radians(M_rise)))
        + (0.020 * math.sin(math.radians(2 * M_rise)))
        + 282.634
    ) % 360

    # Calculate the Sun's right ascension, and adjust angle to be between 0
    # and 360
    RA_rise = (math.degrees(math.atan(0.91764 * math.tan(math.radians(L_rise))))) % 360

    # Right ascension value needs to be in the same quadrant as L
    Lquadrant_rise = (math.floor(L_rise / 90)) * 90
    RAquadrant_rise = (math.floor(RA_rise / 90)) * 90
    RA_rise = RA_rise + (Lquadrant_rise - RAquadrant_rise)

    # Right ascension value needs to be converted into hours
    RA_rise = RA_rise / 15

    # Calculate the Sun's declination
    sinDec_rise = 0.39782 * math.sin(math.radians(L_rise))
    cosDec_rise = math.cos(math.asin(sinDec_rise))

    # Calculate the Sun's local hour angle
    cos_zenith = math.cos(math.radians(zenith))
    radian_lat = math.radians(latitude)
    sin_latitude = math.sin(radian_lat)
    cos_latitude = math.cos(radian_lat)
    cosH_rise = (cos_zenith - (sinDec_rise * sin_latitude)) / (
        cosDec_rise * cos_latitude
    )

    # Finish calculating H and convert into hours
    H_rise = (360 - math.degrees(math.acos(cosH_rise))) / 15

    # Calculate local mean time of rising/setting
    T_rise = H_rise + RA_rise - (0.06571 * t_rise) - 6.622

    # Adjust back to UTC, and keep the time between 0 and 24
    UT_rise = (T_rise - lngHour) % 24

    # Conversion
    h_rise = int(UT_rise)
    m_rise = int(UT_rise % 1 * 60)

    # Create datetime objects with same date, but with hour and minute
    # specified
    rise_dt = datetime.datetime(
        year=year, day=day, month=month, hour=h_rise, minute=m_rise
    )

    return rise_dt
