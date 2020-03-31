import pyart.io.nexrad_common as nexrad


# given bounding latitude and longitude, this method returs all radars in the
#  area
def getRadarNames(minLat, maxLat, minLon, maxLon):
    Radars = []
    for key in nexrad.NEXRAD_LOCATIONS.keys():
        if key[0] == "K":
            lat = float(nexrad.NEXRAD_LOCATIONS[key]["lat"])
            lon = float(nexrad.NEXRAD_LOCATIONS[key]["lon"])
            if lat >= minLat and lat <= maxLat and lon >= minLon and lon <= maxLon:
                Radars.append(key)
    return Radars


# set latitude and longitude bounding box for the data being used
# Returns a list of all radars found in this area
def getEasternRadarNames():
    minLat = 24.5
    maxLat = 48
    minLon = -99
    maxLon = -70
    return getRadarNames(minLat, maxLat, minLon, maxLon)


def getBirdRadarNames():
    birdRadars = [
        "KLCH",
        "KILX",
        "KTBW",
        "KILN",
        "KFFC",
        "KMXX",
        "KVWX",
        "KJGX",
        "KEWX",
        "KLOT",
        "KHTX",
        "KJKL",
        "KLZK",
        "KLVX",
        "KJAX",
        "KHPX",
        "KMRX",
        "KOKX",
        "KLSX",
        "KIND",
        "KEAX",
        "KMOB",
        "KLWX",
        "KDTX",
        "KBOX",
        "KDMX",
        "KFWS",
        "KEVX",
        "KGWX",
        "KCAE",
        "KEOX",
        "KLIX",
        "KBRO",
        "KDOX",
        "KOAX",
        "KINX",
        "KGRR",
        "KDLH",
        "KSRX",
        "KNQA",
        "KRAX",
        "KPOE",
        "KAMX",
        "KPAH",
        "KTLX",
        "KMPX",
        "KICT",
        "KRLX",
        "KLTX",
        "KGRB",
        "KIWX",
        "KHGX",
        "KPBZ",
        "KGRK",
        "KVAX",
        "KFSD",
        "KTLH",
        "KBUF",
        "KMHX",
        "KAKQ",
        "KSGF",
        "KTYX",
        "KFCX",
        "KCLE",
    ]
    return birdRadars


def getRadarLocation(radar):
    return [
        nexrad.NEXRAD_LOCATIONS[radar]["lat"],
        nexrad.NEXRAD_LOCATIONS[radar]["lon"],
    ]


def main():
    # Get the print radars in the eastern part of the united states
    print("Bird Radars", getBirdRadarNames())
    print("eastern radars", getEasternRadarNames())


if __name__ == "__main__":
    main()
