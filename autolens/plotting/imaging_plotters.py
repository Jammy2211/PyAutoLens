from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import plotters
from autolens.plotting import plotter_tools


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

    plot_image_as_subplot = conf.instance.general.get('output', 'plot_imaging_as_subplot', bool)

    if plot_image_as_subplot or ignore_config:

        rows, columns, figsize = plotter_tools.get_subplot_rows_columns_figsize(number_subplots=4)

        plt.figure(figsize=figsize)
        plt.subplot(rows, columns, 1)

        plot_image(image=image, mask=mask, positions=positions, grid=None, as_subplot=True,
                   units=units, kpc_per_arcsec=None, xyticksize=16,
                   norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
                   titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=output_path, output_format=output_format)

        plt.subplot(rows, columns, 2)

        plot_noise_map(image=image, mask=mask, units=units, as_subplot=True,
            kpc_per_arcsec=None, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_format=output_format)

        plt.subplot(rows, columns, 3)

        plot_psf(image=image, units='arcsec', as_subplot=True,
            kpc_per_arcsec=None, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_format=output_format)

        plt.subplot(rows, columns, 4)

        plot_signal_to_noise_map(image=image, mask=mask, as_subplot=True,
            units=units, kpc_per_arcsec=None, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_format=output_format)

        plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                           output_format=output_format)

        plt.close()


def plot_image_individual(image, mask=None, positions=None, output_path=None, output_format='png'):
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
        plot_noise_map(image=image, mask=mask, output_path=output_path, output_format=output_format)

    if plot_imaging_psf:
        plot_psf(image=image, output_path=output_path, output_format=output_format)

    if plot_imaging_signal_to_noise_map:
        plot_signal_to_noise_map(image=image, mask=mask, output_path=output_path, output_format=output_format)


def plot_image(image, mask=None, positions=None, grid=None, as_subplot=False,
               units='arcsec', kpc_per_arcsec=None,
               xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
               title='Observed Image', titlesize=46, xlabelsize=36, ylabelsize=36,
               output_path=None, output_format='show', output_filename='observed_image'):

    plotters.plot_image(image, mask, positions, grid, as_subplot,
                        units, kpc_per_arcsec, xyticksize, norm, norm_min,
                        norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                        titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)


def plot_noise_map(image, mask=None, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                   title='Noise-Map', titlesize=46, xlabelsize=36, ylabelsize=36,
                   output_path=None, output_format='show', output_filename='noise_map'):

    plotters.plot_noise_map(image.noise_map, mask, as_subplot,
                            units, kpc_per_arcsec, xyticksize, norm, norm_min,
                            norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                            titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_psf(image, as_subplot=False,
             units='arcsec', kpc_per_arcsec=None,
             xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
             figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
             title='PSF', titlesize=46, xlabelsize=36, ylabelsize=36,
             output_path=None, output_format='show', output_filename='psf'):

    plotters.plot_psf(image.psf, as_subplot,
                      units, kpc_per_arcsec, xyticksize, norm, norm_min,
                      norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                      titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_signal_to_noise_map(image, mask=None, as_subplot=False,
                             units='arcsec', kpc_per_arcsec=None,
                             xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                             title='Noise-Map', titlesize=46, xlabelsize=36, ylabelsize=36,
                             output_path=None, output_format='show', output_filename='signal_to_noise_map'):

    plotters.plot_signal_to_noise_map(image.signal_to_noise_map, mask, as_subplot,
                        units, kpc_per_arcsec, xyticksize, norm, norm_min,
                        norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                        titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_grid(image, xmin=None, xmax=None, ymin=None, ymax=None,
              output_path=None, output_format='show', output_filename='grid'):

    plotters.plot_grid(grid=image.grid_1d, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_path=output_path,
                       output_format=output_format, output_filename=output_filename)
