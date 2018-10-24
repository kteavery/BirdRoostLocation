import matplotlib.pyplot as plt
import pyart.graph
import pyart.io
from BirdRoostDetection import utils
import os
from PIL import Image

plot_dict = {
    utils.Radar_Products.reflectivity: [-20, 30, None],
    utils.Radar_Products.velocity: [-20, 20, 'coolwarm'],
    utils.Radar_Products.diff_reflectivity: [-4, 8, 'coolwarm'],
    utils.Radar_Products.cc: [.3, .95, 'jet']
}


def visualizeRadardata(radar, save, dualPolarization=False,
                       displayCircles=False):
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
    display = pyart.graph.RadarDisplay(radar)
    if (dualPolarization):
        fig = plt.figure(figsize=(9, 9))
    else:
        fig = plt.figure(figsize=(9, 4.5))
    # display the lowest elevation scan data
    plots = []
    plots.append([utils.Radar_Products.reflectivity, 'Reflectivity_0 (dBZ)', 0])
    plots.append([utils.Radar_Products.velocity, 'Velocity (m/s)', 1])
    if (dualPolarization):
        plots.append([utils.Radar_Products.diff_reflectivity,
                      r'Differential Reflectivity $Z_{DR}$ (dB)', 0])
        plots.append([utils.Radar_Products.cc,
                      r'Correlation Coefficient $\rho_{HV}$', 0])
    ncols = 2
    nrows = len(plots) / 2
    for plotno, plot in enumerate(plots, start=1):
        ax = fig.add_subplot(nrows, ncols, plotno)
        radar_field = plot[0]
        display.plot(utils.pyart_key_dict[radar_field], plot[2], ax=ax,
                     title=plot[1],
                     vmin=plot_dict[radar_field][0],
                     vmax=plot_dict[radar_field][1],
                     cmap=plot_dict[radar_field][2],
                     colorbar_label='',
                     axislabels=(
                         'East-West distance from radar (km)' if plotno == 6
                         else
                         '',
                         'North-South distance from radar (km)' if plotno == 1
                         else ''))
        radius = 300
        display.set_limits((-radius, radius), (-radius, radius), ax=ax)
        display.set_aspect_ratio('equal', ax=ax)
        if (displayCircles):
            display.plot_range_rings(range(100, 350, 100), lw=0.5, col='black',
                                     ax=ax)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()


def visualizeBWRadarData(file_path, img_path, save):
    file = open(file_path, 'r')
    rad = pyart.io.read_nexrad_archive(file.name)
    if (6 <= int(file_path[-1])):
        keys = [utils.Radar_Products.reflectivity,
                utils.Radar_Products.velocity,
                utils.Radar_Products.diff_reflectivity, utils.Radar_Products.cc]
    else:
        keys = [utils.Radar_Products.reflectivity,
                utils.Radar_Products.velocity, ]
    for key in keys:
        if not (save and os.path.exists(img_path.format(key.fullname))):
            fig, ax = plt.subplots(figsize=(3, 3))
            num_sweeps = len(rad.sweep_number['data'])
            for i in range(min(num_sweeps, 3)):
                __plot_ppi(radar=rad, field=key, ax=ax,
                           sweep=i)
            plt.axis('off')

            if save:
                full_img_path = img_path.format(key.fullname)
                img_folder = os.path.dirname(full_img_path)
                if not os.path.isdir(img_folder):
                    os.makedirs(img_folder)
                plt.savefig(full_img_path)
                Image.open(full_img_path).convert('L').save(full_img_path)

            else:
                plt.show()
            plt.close()


def __plot_ppi(radar, field, ax, sweep=0):
    # get data for the plot
    sweep_slice = radar.get_slice(sweep)
    data = radar.fields[utils.pyart_key_dict[field]]['data'][sweep_slice]

    x, y, _ = radar.get_gate_x_y_z(
        sweep, edges=False, filter_transitions=True)
    x = x / 1000.0
    y = y / 1000.0

    cutoff = 1400
    x = x[:, 0:cutoff]
    y = y[:, 0:cutoff]
    data = data[:, 0:cutoff]

    ax.pcolormesh(
        x, y, data, vmin=plot_dict[field][0], vmax=plot_dict[field][1],
        cmap='binary')
