from matplotlib import pyplot as plt

from autolens import conf
from autolens.visualize import array_plotters
from autolens.visualize import util


def plot_image(image, output_path=None, output_filename='images', output_format='show', ignore_config=True):
    """Plot the observed _image of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    image : autolens.imaging.image.Image
        Class containing the _image, noise-mappers and PSF that are to be plotted.
    units : str
        The units the figure is in, which determine the xyticks and xylabels. Options are arcsec | kpc.
    norm : str
        The normalization of the colormap used for plotting. Choose from linear | log | symmetric_log \
        (see matplotlib.colors)
    norm_min : float
        The minimum value in the colormap (see matplotlib.colors). If None, the minimum value in the array is used.
    norm_max : float
        The maximum value in the colormap (see matplotlib.colors). If None, the maximum value in the array is used.
    linthresh : float
        If the symmetric log norm is used, this sets the range within which the colormap is linear \
         (see matplotlib.colors).
    liscale : float
        If the symmetric log norm is used, this stretches the linear range relative to the log range \
        (see matplotlib.colors).
    figsize : (int, int)
        The size the figure is plotted (see matplotlib.pyplot).
    aspect : str
        The aspect ratio of the _image, the default 'auto' scales this to the window size (see matplotlib.pyplot).
    cmap : str
        The colormap style (e.g. 'jet', 'warm', 'binary, see matplotlib.pyplot).
    image_title : str
        The title of the _image.
    noise_map_title : str
        The title of the noise-mappers _image.
    psf_title : str
        The title of the psf _image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fits)
    output_image_filename : str
        The name of the file that the _image is output, if the output_type is a file format (e.g. png, fits)
    output_noise_map_filename : str
        The name of the file that the _image is output, if the output_type is a file format (e.g. png, fits)
    output_psf_filename : str
        The name of the file that the _image is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _image is output. File formats (e.g. png, fits) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plot_image_as_subplot = conf.instance.general.get('output', 'plot_image_as_subplot', bool)

    if plot_image_as_subplot and ignore_config is False:
        plt.figure(figsize=(25, 20))
        plt.subplot(2, 2, 1)

        array_plotters.plot_image(
            image=image, as_subplot=True,
            xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 2, 2)

        array_plotters.plot_noise_map(
            noise_map=image.noise_map, as_subplot=True,
            xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 2, 3)

        array_plotters.plot_psf(
            psf=image.psf, as_subplot=True,
            xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 2, 4)

        array_plotters.plot_signal_to_noise_map(
            signal_to_noise_map=image.signal_to_noise_map, as_subplot=True,
            xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        util.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
        plt.close()


def plot_image_individuals(image, output_path=None, output_format='show'):
    """Plot the observed _image of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    image : autolens.imaging.image.Image
        Class containing the _image, noise-mappers and PSF that are to be plotted.
    units : str
        The units the figure is in, which determine the xyticks and xylabels. Options are arcsec | kpc.
    norm : str
        The normalization of the colormap used for plotting. Choose from linear | log | symmetric_log \
        (see matplotlib.colors)
    norm_min : float
        The minimum value in the colormap (see matplotlib.colors). If None, the minimum value in the array is used.
    norm_max : float
        The maximum value in the colormap (see matplotlib.colors). If None, the maximum value in the array is used.
    linthresh : float
        If the symmetric log norm is used, this sets the range within which the colormap is linear \
         (see matplotlib.colors).
    liscale : float
        If the symmetric log norm is used, this stretches the linear range relative to the log range \
        (see matplotlib.colors).
    figsize : (int, int)
        The size the figure is plotted (see matplotlib.pyplot).
    aspect : str
        The aspect ratio of the _image, the default 'auto' scales this to the window size (see matplotlib.pyplot).
    cmap : str
        The colormap style (e.g. 'jet', 'warm', 'binary, see matplotlib.pyplot).
    image_title : str
        The title of the _image.
    noise_map_title : str
        The title of the noise-mappers _image.
    psf_title : str
        The title of the psf _image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fits)
    output_image_filename : str
        The name of the file that the _image is output, if the output_type is a file format (e.g. png, fits)
    output_noise_map_filename : str
        The name of the file that the _image is output, if the output_type is a file format (e.g. png, fits)
    output_psf_filename : str
        The name of the file that the _image is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _image is output. File formats (e.g. png, fits) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plot_image_image = conf.instance.general.get('output', 'plot_image_image', bool)
    plot_image_noise_map = conf.instance.general.get('output', 'plot_image_noise_map', bool)
    plot_image_psf = conf.instance.general.get('output', 'plot_image_psf', bool)
    plot_image_signal_to_noise_map = \
        conf.instance.general.get('output', 'plot_image_signal_to_noise_map', bool)

    if plot_image_image:
        array_plotters.plot_image(
            image=image, as_subplot=False,
            xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_format=output_format)

    if plot_image_noise_map:
        array_plotters.plot_noise_map(
            noise_map=image.noise_map, as_subplot=False,
            xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_format=output_format)

    if plot_image_psf:
        array_plotters.plot_psf(
            psf=image.psf, as_subplot=False,
            xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_format=output_format)

    if plot_image_signal_to_noise_map:
        array_plotters.plot_signal_to_noise_map(
            signal_to_noise_map=image.signal_to_noise_map, as_subplot=False,
            xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_format=output_format)
