from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import array_plotters


def plot_fitting(fit, output_path=None, output_filename='fit', output_format='show', ignore_config=True):
    plot_fitting_as_subplot = conf.instance.general.get('output', 'plot_fitting_as_subplot', bool)

    if not plot_fitting_as_subplot and ignore_config is False:
        return

    if fit.total_planes == 1:

        if not fit.is_hyper_fit:

            plot_fitting_lens_plane_only(fit, output_path, output_filename, output_format)

        elif fit.is_hyper_fit:

            plot_fitting_hyper_lens_plane_only(fit, output_path, output_filename, output_format)

    elif fit.total_planes == 2:

        if not fit.is_hyper_fit:
            plot_fitting_lens_and_source_planes(fit, output_path, output_filename, output_format)


def plot_fitting_lens_plane_only(fit, output_path=None, output_filename='fit', output_format='show'):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
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
    title : str
        The title of the _image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plt.figure(figsize=(25, 20))
    plt.subplot(2, 2, 1)

    array_plotters.plot_array(
        array=fit.image, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Observed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 2, 2)

    array_plotters.plot_array(
        array=fit.model_image, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 2, 3)

    array_plotters.plot_array(
        array=fit.residuals, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Residuals', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 2, 4)

    array_plotters.plot_array(
        array=fit.chi_squareds, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Chi Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    array_plotters.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                        output_format=output_format)
    plt.close()


def plot_fitting_hyper_lens_plane_only(fit, output_path=None, output_filename='fit', output_format='show'):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
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
    title : str
        The title of the _image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plt.figure(figsize=(25, 20))

    plt.subplot(3, 3, 1)

    array_plotters.plot_array(
        array=fit.image, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Observed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 2)

    array_plotters.plot_array(
        array=fit.model_image, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 3)

    array_plotters.plot_array(
        array=fit.residuals, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Residuals', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 4)

    # If more than one galaxy (and thus contribution map) sum them

    if len(fit.contributions) > 1:
        contributions = sum(fit.contributions)
    else:
        contributions = fit.contributions[0]

    array_plotters.plot_array(
        array=contributions, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Lens-Plane Contributions', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 5)

    array_plotters.plot_array(
        array=fit.chi_squareds, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 6)

    array_plotters.plot_array(
        array=fit.scaled_chi_squareds, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Scaled Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 8)

    array_plotters.plot_array(
        array=fit.noise_map, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Noise Map', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 9)

    array_plotters.plot_array(
        array=fit.scaled_noise_map, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Scaled Noise Map', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    array_plotters.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()


def plot_fitting_lens_and_source_planes(fit, output_path=None, output_filename='fit', output_format='show'):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
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
    title : str
        The title of the _image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plt.figure(figsize=(25, 20))
    plt.subplot(3, 3, 1)

    array_plotters.plot_array(
        array=fit.image, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Observed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 4)

    array_plotters.plot_array(
        array=fit.model_image, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    if fit.model_images_of_planes[0] is not None:
        plt.subplot(3, 3, 5)

        array_plotters.plot_array(
            array=fit.model_images_of_planes[0], grid=None, as_subplot=True,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Lens Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 6)

    array_plotters.plot_array(
        array=fit.model_images_of_planes[1], grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Source Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 7)

    array_plotters.plot_array(
        array=fit.plane_images[1], grid=fit.plane_grids[1], as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Source-Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 8)

    array_plotters.plot_array(
        array=fit.residuals, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Residuals', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 9)

    array_plotters.plot_array(
        array=fit.chi_squareds, grid=None, as_subplot=True,
        xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    array_plotters.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()


def plot_fitting_individuals(fit, output_path=None, output_format='show'):
    
    if fit.total_planes == 1:

        if not fit.is_hyper_fit:

            plot_fitting_individuals_lens_plane_only(fit, output_path, output_format)

        elif fit.is_hyper_fit:

            plot_fitting_individuals_hyper_lens_plane_only(fit, output_path, output_format)

    elif fit.total_planes == 2:

        if not fit.is_hyper_fit:
            plot_fitting_individuals_lens_and_source_planes(fit, output_path, output_format)


def plot_fitting_individuals_lens_plane_only(fitting, output_path=None, output_format='show'):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fitting : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
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
    title : str
        The title of the _image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plot_fitting_model_image = conf.instance.general.get('output', 'plot_fitting_model_image', bool)
    plot_fitting_residuals = conf.instance.general.get('output', 'plot_fitting_residuals', bool)
    plot_fitting_chi_squareds = conf.instance.general.get('output', 'plot_fitting_chi_squareds', bool)

    if plot_fitting_model_image:
        array_plotters.plot_array(
            array=fitting.model_image, grid=None, as_subplot=False,
            xticks=fitting.image.xticks, yticks=fitting.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='model_image', output_format=output_format)

    if plot_fitting_residuals:

        array_plotters.plot_array(
            array=fitting.residuals, grid=None, as_subplot=False,
            xticks=fitting.image.xticks, yticks=fitting.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Residuals', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='residuals', output_format=output_format)

    if plot_fitting_chi_squareds:

        array_plotters.plot_array(
            array=fitting.chi_squareds, grid=None, as_subplot=False,
            xticks=fitting.image.xticks, yticks=fitting.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Chi-Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='chi_squareds', output_format=output_format)


def plot_fitting_individuals_hyper_lens_plane_only(fit, output_path=None, output_format='show'):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
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
    title : str
        The title of the _image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plot_fitting_model_image = conf.instance.general.get('output', 'plot_fitting_model_image', bool)
    plot_fitting_residuals = conf.instance.general.get('output', 'plot_fitting_residuals', bool)
    plot_fitting_chi_squareds = conf.instance.general.get('output', 'plot_fitting_chi_squareds', bool)
    plot_fitting_contributions = conf.instance.general.get('output', 'plot_fitting_contributions', bool)
    plot_fitting_scaled_chi_squareds = conf.instance.general.get('output', 'plot_fitting_scaled_chi_squareds', bool)
    plot_fitting_scaled_noise_map = conf.instance.general.get('output', 'plot_fitting_scaled_noise_map', bool)

    if plot_fitting_model_image:
        array_plotters.plot_array(
            array=fit.model_image, grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Observed Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='model_image', output_format=output_format)

    if plot_fitting_residuals:
        array_plotters.plot_array(
            array=fit.residuals, grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Residuals', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='residuals', output_format=output_format)

    if plot_fitting_chi_squareds:
        array_plotters.plot_array(
            array=fit.chi_squareds, grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Chi Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='chi_squareds', output_format=output_format)

    if plot_fitting_contributions:

        if len(fit.contributions) > 1:
            contributions = sum(fit.contributions)
        else:
            contributions = fit.contributions[0]

        array_plotters.plot_array(
            array=contributions, grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Lens-Plane Contributions', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='lens_plane_contributions', output_format=output_format)

    if plot_fitting_scaled_noise_map:
        array_plotters.plot_array(
            array=fit.scaled_noise_map, grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Scaled Noise Map', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='scaled_noise_map', output_format=output_format)

    if plot_fitting_scaled_chi_squareds:
        array_plotters.plot_array(
            array=fit.scaled_chi_squareds, grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Scaled Chi-Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path,output_filename='scaled_chi_squareds', output_format=output_format)


def plot_fitting_individuals_lens_and_source_planes(fit, output_path=None, output_format='show'):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
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
    title : str
        The title of the _image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plot_fitting_model_image = conf.instance.general.get('output', 'plot_fitting_model_image', bool)
    plot_fitting_lens_model_image = conf.instance.general.get('output', 'plot_fitting_lens_model_image', bool)
    plot_fitting_source_model_image = conf.instance.general.get('output', 'plot_fitting_source_model_image', bool)
    plot_fitting_plane_image = conf.instance.general.get('output', 'plot_fitting_plane_image', bool)
    plot_fitting_residuals = conf.instance.general.get('output', 'plot_fitting_residuals', bool)
    plot_fitting_chi_squareds = conf.instance.general.get('output', 'plot_fitting_chi_squareds', bool)

    if plot_fitting_model_image:
        array_plotters.plot_array(
            array=fit.model_image, grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='model_image', output_format=output_format)

    if plot_fitting_lens_model_image:
        array_plotters.plot_array(
            array=fit.model_images_of_planes[0], grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Lens Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='lens_model_image', output_format=output_format)

    if plot_fitting_source_model_image:
        array_plotters.plot_array(
            array=fit.model_images_of_planes[1], grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Source Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='source_model_image', output_format=output_format)

    if plot_fitting_plane_image:
        array_plotters.plot_array(
            array=fit.plane_images[1], grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Source-Plane Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='source_plane', output_format=output_format)

    if plot_fitting_residuals:
        array_plotters.plot_array(
            array=fit.residuals, grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Residuals', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='residuals', output_format=output_format)

    if plot_fitting_chi_squareds:
        array_plotters.plot_array(
            array=fit.chi_squareds, grid=None, as_subplot=False,
            xticks=fit.image.xticks, yticks=fit.image.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Chi-Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='chi_squareds', output_format=output_format)


# def plot_fitting_hyper_arrays(fitting, output_path=None, output_format='show'):
#
#     plot_fitting_hyper_arrays = conf.instance.general.get('output', 'plot_fitting_hyper_arrays', bool)
#
#     if plot_fitting_hyper_arrays:
#
#         array_plotters.plot_array(fitting.unmasked_model_image, output_filename='unmasked_model_image',
#                                         output_path=output_path, output_format=output_format)
#
#         for i, unmasked_galaxy_model_image in enumerate(fitting.unmasked_model_images_of_galaxies):
#             array_plotters.plot_array(unmasked_galaxy_model_image,
#                                             output_filename='unmasked_galaxy_image_' + str(i),
#                                             output_path=output_path, output_format=output_format)
