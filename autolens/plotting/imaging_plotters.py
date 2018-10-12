from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import plotters


def plot_image_subplot(image, mask=None, positions=None, units='arcsec', output_path=None, output_filename='images',
                       output_format='show', ignore_config=True, figsize=(25, 20)):
    """Plot the observed _image of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    figsize
    ignore_config
    output_filename
    units
    positions
    image : autolens.imaging.image.Image
        Class containing the _image, noise-mappers and PSF that are to be plotted.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _image is output. File formats (e.g. png, fits) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plot_image_as_subplot = conf.instance.general.get('output', 'plot_image_as_subplot', bool)

    if plot_image_as_subplot or ignore_config:

        plt.figure(figsize=figsize)
        plt.subplot(2, 2, 1)

        plot_image(as_subplot=True,
            image=image, mask=mask, positions=positions, grid=None, units=units, kpc_per_arcsec=None, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_format=output_format)

        plt.subplot(2, 2, 2)

        plot_noise_map(as_subplot=True,
            noise_map=image.noise_map, mask=mask, units=units, kpc_per_arcsec=None, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_format=output_format)

        plt.subplot(2, 2, 3)

        plot_psf(as_subplot=True,
            psf=image.psf, units='arcsec', kpc_per_arcsec=None, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_format=output_format)

        plt.subplot(2, 2, 4)

        plot_signal_to_noise_map(as_subplot=True, signal_to_noise_map=image.signal_to_noise_map, mask=mask,
            units=units, kpc_per_arcsec=None, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_format=output_format)

        plotters.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

        plt.close()


def plot_image_individuals(image, mask=None, positions=None, output_path=None, output_format='png'):
    """Plot the observed _image of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    figsize
    plot_signal_to_noise_map
    ignore_config
    plot_psf
    plot_noise_map
    plot_image
    positions
    image : autolens.imaging.image.Image
        Class containing the _image, noise-mappers and PSF that are to be plotted.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _image is output. File formats (e.g. png, fits) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plot_imaging_image = conf.instance.general.get('output', 'plot_imaging_image', bool)
    plot_imaging_noise_map = conf.instance.general.get('output', 'plot_imaging_noise_map', bool)
    plot_imaging_psf = conf.instance.general.get('output', 'plot_imaging_psf', bool)
    plot_imaging_signal_to_noise_map = conf.instance.general.get('output', 'plot_imaging_signal_to_noise_map', bool)

    if plot_imaging_image:
        plot_image(image=image, mask=mask, positions=positions, output_path=output_path, output_format=output_format)

    if plot_imaging_noise_map:
        plot_noise_map(noise_map=image.noise_map, mask=mask, output_path=output_path, output_format=output_format)

    if plot_imaging_psf:
        plot_psf(psf=image.psf, output_path=output_path, output_format=output_format)

    if plot_imaging_signal_to_noise_map:
        plot_signal_to_noise_map(signal_to_noise_map=image.signal_to_noise_map, mask=mask, output_path=output_path,
                                 output_format=output_format)


def plot_image(image, mask=None, positions=None, grid=None,
               units='arcsec', kpc_per_arcsec=None,
               xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
               title='Observed Image', titlesize=46, xlabelsize=36, ylabelsize=36,
               output_path=None, output_format='show', output_filename='observed_image', as_subplot=False):

    if positions is not None:
        positions = list(map(lambda pos: image.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=pos), positions))

    plotters.plot_array(image, as_subplot, figsize, aspect, cmap, norm, norm_max, norm_min, linthresh, linscale)
    plotters.set_title(title, titlesize)
    plotters.set_xy_labels_and_ticks(image.shape, units, kpc_per_arcsec, image.xticks, image.yticks, xlabelsize,
                                     ylabelsize, xyticksize)
    plotters.set_colorbar(cb_ticksize)
    plotters.plot_points(positions)
    plotters.plot_mask(mask)
    plotters.plot_grid(grid)
    plotters.output_array(image, output_path, output_filename, output_format)
    plt.close()

def plot_noise_map(noise_map, mask=None,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                   title='Noise-Map', titlesize=46, xlabelsize=36, ylabelsize=36,
                   output_path=None, output_format='show', output_filename='noise_map', as_subplot=False):


    plotters.plot_array(noise_map, as_subplot, figsize, aspect, cmap, norm, norm_max, norm_min, linthresh, linscale)
    plotters.set_title(title, titlesize)
    plotters.set_xy_labels_and_ticks(noise_map.shape, units, kpc_per_arcsec, noise_map.xticks, noise_map.yticks,
                                     xlabelsize, ylabelsize, xyticksize)
    plotters.set_colorbar(cb_ticksize)
    plotters.plot_mask(mask)
    plotters.output_array(noise_map, output_path, output_filename, output_format)
    plt.close()

def plot_psf(psf,
             units='arcsec', kpc_per_arcsec=None,
             xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
             figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
             title='PSF', titlesize=46, xlabelsize=36, ylabelsize=36,
             output_path=None, output_format='show', output_filename='psf', as_subplot=False):


    plotters.plot_array(psf, as_subplot, figsize, aspect, cmap, norm, norm_max, norm_min, linthresh, linscale)
    plotters.set_title(title, titlesize)
    plotters.set_xy_labels_and_ticks(psf.shape, units, kpc_per_arcsec, psf.xticks, psf.yticks,
                                     xlabelsize, ylabelsize, xyticksize)
    plotters.set_colorbar(cb_ticksize)
    plotters.output_array(psf, output_path, output_filename, output_format)
    plt.close()

def plot_signal_to_noise_map(signal_to_noise_map, mask=None,
                             units='arcsec', kpc_per_arcsec=None,
                             xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                             title='Noise-Map', titlesize=46, xlabelsize=36, ylabelsize=36,
                             output_path=None, output_format='show', output_filename='signal_to_noise_map',
                             as_subplot=False):


    plotters.plot_array(signal_to_noise_map, as_subplot, figsize, aspect, cmap, norm, norm_max, norm_min, linthresh,
                        linscale)
    plotters.set_title(title, titlesize)
    plotters.set_xy_labels_and_ticks(signal_to_noise_map.shape, units, kpc_per_arcsec,
                                     signal_to_noise_map.xticks, signal_to_noise_map.yticks,
                                     xlabelsize, ylabelsize, xyticksize)
    plotters.set_colorbar(cb_ticksize)
    plotters.plot_mask(mask)
    plotters.output_array(signal_to_noise_map, output_path, output_filename, output_format)
    plt.close()

def plot_grid(grid, xmin=None, xmax=None, ymin=None, ymax=None,
              output_path=None, output_format='show', output_filename='grid'):

    plt.figure()
    plt.scatter(y=grid[:, 0], x=grid[:, 1], marker='.')
    plotters.set_title(title='Grid', titlesize=36)
    plt.ylabel('y (arcsec)', fontsize=18)
    plt.xlabel('x (arcsec)', fontsize=18)
    plt.tick_params(labelsize=20)
    if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
        plt.axis([xmin, xmax, ymin, ymax])
    plotters.output_array(None, output_path, output_filename, output_format)
    plt.close()