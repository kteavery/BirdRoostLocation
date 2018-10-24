""" Get NEXRAD radar data from Amazon Web Services."""
import BirdRoostDetection.LoadSettings as settings
import tempfile
import boto
import pyart.graph
import pyart.io
import os
import datetime


def ConnectToAWS():
    """Connet to the Amazon server.

    Connect to the Amazon server. In order to call ths method you will need to
    have your amazon server credentials save on your computer in a .boto file.
    For more instructions see http://boto.cloudhackers.com/en/latest/s3_tut.html

    Returns:
        A connection to Amazon's S3
    """
    conn = boto.connect_s3()  # connect to aws
    return conn


def GetNexradBucket(conn):
    """Get access to the NOAA NEXRAD level 2 radar data bucket.

    Args:
        conn: A connection to Amazon's S3

    Returns:
        An instantiated NOAA NEXRAD level 2 radar data bucket.
    """
    bucket = conn.get_bucket('noaa-nexrad-level2', validate=False)
    return bucket


# Get the name of the buck for a year, month, day and radar
def getBucketName(year, month, day, radar):
    """ Get the name of a specific bucket where radar data is stored.

    Args:
        year: Year as an integer.
        month: Month as an integer.
        day: Day as an integer.
        radar: The 4 letter name of the radar, a string.

    Returns:
        The bucket name as a string (e.g. YYYY/MM/DD/KLMN/).
    """
    # Get the name of the buck for a year, month, day and radar
    try:
        return "%04d/%02d/%02d/%s/" % (year, month, day, radar)
    except:
        return "%s/%s/%s/%s/" % (year, month, day, radar)


def getFileNamesFromBucket(bucket, bucketName):
    """ Get the name of every file given a bucket.

    Args:
        bucket: A NOAA NEXRAD level 2 radar data bucket
        bucketName: The bucket name as a string (e.g. YYYY/MM/DD/KLMN/).

    Returns:
        A list of strings. Each string is formatted as follows :
        'YYYY/MM/DD/KLMN/KLMNYYYYMMDD_HHMMSS_V06.gz' or
        'YYYY/MM/DD/KLMN/KLMNYYYYMMDD_HHMMSS_V03.gz'
    """
    names = []
    for key in bucket.list(bucketName, "/"):
        location = key.name.encode('utf-8')
        names.append(location)
    return names


def downloadDataFromBucket(bucket, fileName):
    """ Download a single NEXRAD radar datum.

    Args:
        bucket: A NOAA NEXRAD level 2 radar data bucket.
        fileName: A filename formatted as follows :
            'YYYY/MM/DD/KLMN/KLMNYYYYMMDD_HHMMSS_V06.gz' or
            'YYYY/MM/DD/KLMN/KLMNYYYYMMDD_HHMMSS_V03.gz'

    Returns:
        Temporary version of NEXRAD file.
    """
    s3key = bucket.get_key(fileName)
    localfile = tempfile.NamedTemporaryFile()
    s3key.get_contents_to_filename(localfile.name)
    return localfile


def getFileNameCloseToDatetime(files, radar, roost_date):
    """Get the filename and index of the AWS file closes to datetime.

    Given a list of files with AWS data, a radar, and a roost_date, find the
    file in the list closest to the roost date. When going from Mosaic data
    to AWS date our roost time may be few seconds or a few minutes off since
    not all radars scans have the same date. This method will allow us to ignore
    small time differences between two times.

    Args:
        files: A list of AWS radar file names.
        radar: The 4 letter name of the radar, a string.
        roost_date: A datetime object, the timestamp we are looking for in the
            file list.

    Returns:
        string, integer. A single AWSFile name, as well as the index where it
        was found.
    """
    min_dif = datetime.timedelta.max
    AWSFile = None
    i = 0
    for index, f in enumerate(files):
        base_f = os.path.basename(f)

        if base_f[0:4] == radar and base_f[13:15] > '08' and base_f[
                                                             13:15] < '12':

            radar_date = datetime.datetime.strptime(base_f[4:19],
                                                    '%Y%m%d_%H%M%S')
            dif = max(roost_date - radar_date, radar_date - roost_date)
            if (min_dif > dif):
                min_dif = dif
                AWSFile = f
                i = index
            else:
                break

    return AWSFile, i


def main():
    """Example of how to use methods of this class."""

    conn = ConnectToAWS()
    bucket = GetNexradBucket(conn)

    # Bucket for a single Radar, single day.
    bucketName = getBucketName(2015, 7, 4, 'KMOB')

    # Get list of all files from bucket
    fileNames = getFileNamesFromBucket(bucket, bucketName)
    fileName = fileNames[0]  # Select file from list
    file = downloadDataFromBucket(bucket, fileName)
    radar = pyart.io.read_nexrad_archive(file.name)
    print radar

    # Use this format if you already know the exact name of the desired file
    fileName = '2015/07/04/KMOB/KMOB20150704_111944_V06.gz'
    file = downloadDataFromBucket(bucket, fileName)
    radar = pyart.io.read_nexrad_archive(file.name)
    print radar

    conn.close()


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    main()
