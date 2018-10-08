from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import array_plotters


def plot_image(image,
               positions=None,
               units='arcsec',
               output_path=None,
               output_filename='images',
               output_format='show',
               ignore_config=True,
               figsize=(25, 20)):
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

        if positions is not None:
            positions = list(map(lambda pos:
                                 image.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=pos),
                                 positions))

        array_plotters.plot_array(
            array=image, points=positions, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=None,
            xticks=image.xticks, yticks=image.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Observed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 2, 2)

        array_plotters.plot_array(
            array=image.noise_map, points=None, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=None,
            xticks=image.xticks, yticks=image.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Noise Map', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 2, 3)

        array_plotters.plot_array(
            array=image.psf, points=None, grid=None, as_subplot=True,
            units='arcsec', kpc_per_arcsec=None,
            xticks=image.psf.xticks(image.pixel_scales), yticks=image.psf.yticks(image.pixel_scales), xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='PSF', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 2, 4)

        array_plotters.plot_array(
            array=image.signal_to_noise_map, points=None, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=None,
            xticks=image.xticks, yticks=image.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Signal To Noise Map', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        array_plotters.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                            output_format=output_format)
        plt.close()


def plot_image_individuals(image,
                           positions=None,
                           plot_image=False,
                           plot_noise_map=False,
                           plot_psf=False,
                           plot_signal_to_noise_map=False,
                           output_path=None,
                           output_format='show',
                           ignore_config=True,
                           figsize=(20, 15)):
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

    if not ignore_config:
        plot_image = conf.instance.general.get('output', 'plot_imaging_image', bool)
        plot_noise_map = conf.instance.general.get('output', 'plot_imaging_noise_map', bool)
        plot_psf = conf.instance.general.get('output', 'plot_imaging_psf', bool)
        plot_signal_to_noise_map = conf.instance.general.get('output', 'plot_imaging_signal_to_noise_map', bool)

    if plot_image:
        array_plotters.plot_array(
            array=image, points=positions, grid=None, as_subplot=False,
            units='arcsec', kpc_per_arcsec=None,
            xticks=image.xticks, yticks=image.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=figsize, aspect='auto', cmap='jet', cb_ticksize=20,
            title='Observed Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='observed_image', output_format=output_format)

    if plot_noise_map:
        array_plotters.plot_array(
            array=image.noise_map, points=None, grid=None, as_subplot=False,
            units='arcsec', kpc_per_arcsec=None,
            xticks=image.xticks, yticks=image.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=figsize, aspect='auto', cmap='jet', cb_ticksize=20,
            title='Noise-Map', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='noise_map', output_format=output_format)

    if plot_psf:
        array_plotters.plot_array(
            array=image.psf, points=None, grid=None, as_subplot=False,
            units='arcsec', kpc_per_arcsec=None,
            xticks=image.psf.xticks(image.pixel_scales), yticks=image.psf.yticks(image.pixel_scales), xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=figsize, aspect='auto', cmap='jet', cb_ticksize=20,
            title='PSF', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='psf', output_format=output_format)

    if plot_signal_to_noise_map:
        array_plotters.plot_array(
            array=image.signal_to_noise_map, points=None, grid=None, as_subplot=False,
            units='arcsec', kpc_per_arcsec=None,
            xticks=image.xticks, yticks=image.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=figsize, aspect='auto', cmap='jet', cb_ticksize=20,
            title='Signal To Noise Map', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='signal_to_noise_map', output_format=output_format)


def plot_grid(grid, xmin=None, xmax=None, ymin=None, ymax=None):
    plt.figure()
    plt.scatter(x=grid[:, 0], y=grid[:, 1], marker='.')
    array_plotters.set_title(title='Grid', titlesize=36)
    plt.xlabel('x (arcsec)', fontsize=36)
    plt.ylabel('y (arcsec)', fontsize=36)
    plt.tick_params(labelsize=40)
    if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
        plt.axis([xmin, xmax, ymin, ymax])
    plt.show()
    plt.close()
