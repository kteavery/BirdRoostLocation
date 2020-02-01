
import matplotlib.pyplot as plt
import numpy as np
import pyart.io.nexrad_common as nexrad
from mpl_toolkits.basemap import Basemap
import pandas

import BirdRoostLocation.Analysis.ListRadars as ListRadars


def addRadarsToPlot(title, radLocation, roostLocations, radars):
    plt.figure(figsize=(11, 8))

    lat = [row[0] for row in radLocation]
    lon = [row[1] for row in radLocation]

    # Set bounds for a map of the united states using lat and lon bounds for
    # bird radars
    map = Basemap(projection="cyl", resolution="l", llcrnrlon=np.min(lon) - 1,
                  urcrnrlon=np.max(lon) + 1,
                  llcrnrlat=np.min(lat) - 1, urcrnrlat=np.max(lat) + 1)

    # Draw the map
    map.drawcoastlines(linewidth=0.5)
    map.drawcountries()
    map.drawstates(linewidth=0.7, color="lightgray")

    plt.title(title)

    x0, y0 = map([row[1] for row in roostLocations[0]],
               [row[0] for row in roostLocations[0]])
    x1, y1 = map([row[1] for row in roostLocations[1]],
               [row[0] for row in roostLocations[1]])
    map.plot(x1, y1, 'ro', markersize=.15, alpha=0.5)
    map.plot(x0, y0, 'co', markersize=.15, alpha=0.5)

    x, y = map(lon, lat)
    for label, xpt, ypt in zip(radars, x, y):
        plt.text(xpt + .25, ypt + .25, label, fontsize=11)
    map.plot(x, y, 'bo', markersize=5, label="Radar Location")
    map.plot(0, 0, 'co', markersize=5, label="Dual-pol Roost Location")
    map.plot(0, 0, 'ro', markersize=5, label="Single-pol Roost Location")
    plt.legend(fontsize=11)
    
    plt.show()


def DrawMapShowingRadars():
    labels = pandas.read_csv(
        '/Users/Kate/workspace/BirdRoostLocation/MLData/true_ml_relabels_polar.csv')
    birdRadars = list(set(labels['radar']))[1:]

    # this is a list of the radars that the OU Radar Aeroecology has found
    # Purple Martins at
    
    # birdRadars = ['KAMX', 'KBRO', 'KDOX', 'KGRK', 'KJAX', 'KHGX', 'KLCH',
    #               'KLIX', 'KMLB', 'KMOB']

    # create an array of all radar locations where purple martins roost
    birdLocation = []
    for key in birdRadars:
        birdLocation.append([nexrad.NEXRAD_LOCATIONS[key]['lat'],
                             nexrad.NEXRAD_LOCATIONS[key]['lon']])

    allRadars = ListRadars.getBirdRadarNames()
    # Create an array of all radar location in the eastern United states
    radLocation = []
    for key in allRadars:
        radLocation.append([nexrad.NEXRAD_LOCATIONS[key]['lat'],
                            nexrad.NEXRAD_LOCATIONS[key]['lon']])

    x = pandas.read_csv(
        '/Users/Kate/workspace/BirdRoostLocation/MLData/true_ml_relabels_polar.csv')

    roostLocationV06 = []
    roostLocationV03 = []
    for i in range(len(x)):
        isV06 = ('V06' in x['AWS_file'][i], x['radar'][i])[0]
        
        if isV06:
            roostLocationV06.append([x['lat'][i], x['lon'][i]])
        else:
            roostLocationV03.append([x['lat'][i], x['lon'][i]])

    addRadarsToPlot('Radar Roost Labels in Dataset',
                    birdLocation, [roostLocationV06, roostLocationV03],
                    birdRadars)


def main():
    DrawMapShowingRadars()


if __name__ == "__main__": main()