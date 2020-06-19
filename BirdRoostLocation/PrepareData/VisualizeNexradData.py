import matplotlib.pyplot as plt
import pyart.graph
import pyart.io
from BirdRoostLocation import utils
from mpl_toolkits.basemap import Basemap
import os
from PIL import Image
import math

import BirdRoostLocation.PrepareData.PyartConfig as pyart_config

vel = pyart_config.DEFAULT_FIELD_LIMITS.get("velocity")

plot_dict = {
    utils.Radar_Products.reflectivity: [-10, 30, "pyart_NWSRef"],
    utils.Radar_Products.velocity: [vel()[0], vel()[1], "pyart_BuDRd18"],
    utils.Radar_Products.diff_reflectivity: [-1, 8, "pyart_RefDiff"],
    utils.Radar_Products.cc: [0.5, 1.05, "pyart_RefDiff"],
}


def visualizeRadarData(
    radar, save, dualPolarization=False, displayCircles=False, nexrads=[], points=[]
):
    """Visualize the LDM2 radar data. Either display or save resulting image.

    This method was modified from code found at the following website :
    https://eng.climate.com/2015/10/27/how-to-read-and-display-nexrad-on-aws
    -using-python/. If the image is dual polarization it will save out
    reflectivity and velocity, otherwise it will also save differential
    reflectivity and correlation coefficient. These later radar products
    are only available in the dual polarization radar upgrades from 2012-2013.

    Args:
        radar:
        save:
        dualPolarization:
        displayCircles:
    """
    # load custom config file with smaller reflectivity range
    # http://arm-doe.github.io/pyart-docs-travis/dev_reference/generated/
    # pyart.config.load_config.html
    pyart.config.load_config(
        filename=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "PyartConfig.py"
        )
    )

    display = pyart.graph.RadarMapDisplay(radar)
    x, y = display._get_x_y(0, True, None)
    if dualPolarization:
        fig = plt.figure(figsize=(9, 9))
    else:
        fig = plt.figure(figsize=(9, 4.5))

    # display the lowest elevation scan data
    plots = []
    plots.append([utils.Radar_Products.reflectivity, "Reflectivity_0 (dBZ)", 0])
    plots.append([utils.Radar_Products.velocity, "Velocity (m/s)", 1])
    if dualPolarization:
        plots.append(
            [
                utils.Radar_Products.diff_reflectivity,
                r"Differential Reflectivity $Z_{DR}$ (dB)",
                0,
            ]
        )
        plots.append(
            [utils.Radar_Products.cc, r"Correlation Coefficient $\rho_{HV}$", 0]
        )
    ncols = 2
    nrows = len(plots) / 2
    for plotno, plot in enumerate(plots, start=1):
        ax = fig.add_subplot(nrows, ncols, plotno)
        radar_field = plot[0]
        display.plot(
            utils.pyart_key_dict[radar_field],
            plot[2],
            ax=ax,
            title=plot[1],
            vmin=plot_dict[radar_field][0],
            vmax=plot_dict[radar_field][1],
            cmap=plot_dict[radar_field][2],
            colorbar_label="",
            axislabels=(
                "East-West distance from radar (km)" if plotno == 6 else "",
                "North-South distance from radar (km)" if plotno == 1 else "",
            ),
        )
        radius = 300
        display.set_limits((-radius, radius), (-radius, radius), ax=ax)
        display.set_aspect_ratio("equal", ax=ax)
        if displayCircles:
            display.plot_range_rings(
                list(range(100, 350, 100)), lw=0.5, col="black", ax=ax
            )
        if points != []:
            print("LAT")
            print(nexrads[0])
            print(nexrads[0] - 1 / (300 * 110.574))
            print(nexrads[0] + 1 / (300 * 110.574))
            print("LON")
            print(nexrads[1])
            print(
                nexrads[1] - 1 / (111.320 * math.cos(nexrads[0] + 1 / (300 * 110.574)))
            )
            print(
                nexrads[1] + 1 / (111.320 * math.cos(nexrads[0] + 1 / (300 * 110.574)))
            )
            display.basemap = Basemap(
                projection="lcc",
                lon_0=nexrads[1],
                lat_0=nexrads[0],
                llcrnrlat=nexrads[0] - 1.25,
                llcrnrlon=nexrads[1] - 1.5,
                urcrnrlat=nexrads[0] + 1.25,
                urcrnrlon=nexrads[1] + 1.5,
                resolution="h",
            )
            x0, y0 = display.basemap(nexrads[1], nexrads[0])
            glons, glats = display.basemap(
                (x0 + x * 1000.0), (y0 + y * 1000.0), inverse=True
            )

            # m.scatter(lon0,lat0,marker='o',s=20,color='k',ax=ax,latlon=True)
            display.plot_point(points[0][0], points[0][1], symbol="ro")
            display.plot_point(points[1][0], points[1][1], symbol="bo")
    # print("SAVE")
    # print(save)
    # if save:
    # print("SAVE")
    # print(save)
    plt.savefig(save)
    # else:
    # plt.show()
    plt.close()


def __plot_ppi(radar, field, ax, sweep=0):
    # get data for the plot
    sweep_slice = radar.get_slice(sweep)
    data = radar.fields[utils.pyart_key_dict[field]]["data"][sweep_slice]

    x, y, _ = radar.get_gate_x_y_z(sweep, edges=False, filter_transitions=True)
    x = x / 1000.0
    y = y / 1000.0

    cutoff = 1400
    x = x[:, 0:cutoff]
    y = y[:, 0:cutoff]
    data = data[:, 0:cutoff]
    # print("X, Y, DATA")
    # print(x)
    # print(y)
    # print(data)

    ax.pcolormesh(
        x, y, data, vmin=plot_dict[field][0], vmax=plot_dict[field][1], cmap="binary"
    )
