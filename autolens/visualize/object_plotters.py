from autolens.visualize import array_plotters
from autolens.lensing import fitting
from matplotlib import pyplot as plt
from autolens import exc

# TODO : Provide optiono so these plots the 3 images of all results on a sub-plot simulatnaouesly

def plot_image(image, output_path=None, output_format='show'):
    """Plot the observed image of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    image : autolens.imaging.image.Image
        Class containing the image, noise-mappers and PSF that are to be plotted.
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
        The aspect ratio of the image, the default 'auto' scales this to the window size (see matplotlib.pyplot).
    cmap : str
        The colormap style (e.g. 'jet', 'warm', 'binary, see matplotlib.pyplot).
    image_title : str
        The title of the image.
    noise_map_title : str
        The title of the noise-mappers image.
    psf_title : str
        The title of the psf image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the image is output if the output_type is a file format (e.g. png, fits)
    output_image_filename : str
        The name of the file that the image is output, if the output_type is a file format (e.g. png, fits)
    output_noise_map_filename : str
        The name of the file that the image is output, if the output_type is a file format (e.g. png, fits)
    output_psf_filename : str
        The name of the file that the image is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the image is output. File formats (e.g. png, fits) output the image to harddisk. 'show' displays the image \
        in the python interpreter window.
    """

    array_plotters.plot_image(
        image=image, as_subplot=False, 
        xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=40,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
        title='Observed Image', titlesize=46, xlabelsize=36, ylabelsize=36,
        output_path=output_path, output_filename='observed_image', output_format=output_format)

    array_plotters.plot_noise_map(
        noise_map=image.noise_map, as_subplot=False, 
        xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=40,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
        title='Noise Map', titlesize=46, xlabelsize=36, ylabelsize=36,
        output_path=output_path, output_filename='noise_map', output_format=output_format)

    array_plotters.plot_psf(
        psf=image.psf, as_subplot=False, 
        xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=40,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
        title='PSF', titlesize=46, xlabelsize=36, ylabelsize=36,
        output_path=output_path, output_filename='psf', output_format=output_format)

    array_plotters.plot_signal_to_noise_map(
        signal_to_noise_map=image.signal_to_noise_map, as_subplot=False, 
        xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=40,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
        title='PSF', titlesize=46, xlabelsize=36, ylabelsize=36,
        output_path=output_path, output_filename='psf', output_format=output_format)

def plot_image_as_subplot(image, output_path=None, output_filename='images', output_format='show'):
    """Plot the observed image of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    image : autolens.imaging.image.Image
        Class containing the image, noise-mappers and PSF that are to be plotted.
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
        The aspect ratio of the image, the default 'auto' scales this to the window size (see matplotlib.pyplot).
    cmap : str
        The colormap style (e.g. 'jet', 'warm', 'binary, see matplotlib.pyplot).
    image_title : str
        The title of the image.
    noise_map_title : str
        The title of the noise-mappers image.
    psf_title : str
        The title of the psf image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the image is output if the output_type is a file format (e.g. png, fits)
    output_image_filename : str
        The name of the file that the image is output, if the output_type is a file format (e.g. png, fits)
    output_noise_map_filename : str
        The name of the file that the image is output, if the output_type is a file format (e.g. png, fits)
    output_psf_filename : str
        The name of the file that the image is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the image is output. File formats (e.g. png, fits) output the image to harddisk. 'show' displays the image \
        in the python interpreter window.
    """

    plt.figure(figsize=(25, 20))
    plt.subplot(2, 2, 1)

    array_plotters.plot_image(
        image=image, as_subplot=True,
        xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=18,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Observed Image', titlesize=24, xlabelsize=20, ylabelsize=20,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 2, 2)

    array_plotters.plot_noise_map(
        noise_map=image.noise_map, as_subplot=True,
        xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=18,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Noise Map', titlesize=24, xlabelsize=20, ylabelsize=20,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 2, 3)

    array_plotters.plot_psf(
        psf=image.psf, as_subplot=True,
        xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=18,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='PSF', titlesize=24, xlabelsize=20, ylabelsize=20,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 2, 4)

    array_plotters.plot_signal_to_noise_map(
        signal_to_noise_map=image.signal_to_noise_map, as_subplot=True,
        xticks=image.xticks, yticks=image.yticks, units='arcsec', xyticksize=18,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Signal-To-Noise Map', titlesize=24, xlabelsize=20, ylabelsize=20,
        output_path=output_path, output_filename=None, output_format=output_format)

    if output_format is 'show':
        plt.show()
    elif output_format is 'png':
        plt.savefig(output_path + output_filename + '.png', bbox_inches='tight')
    elif output_format is 'fits':
        raise exc.VisualizeException('You cannot output a subplots with format .fits')

    plt.close()

def plot_results(results, output_path=None, output_format='show'):
    """Plot the model image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    results : autolens.lensing.fitting.Fitter
        Class containing fit between the model image and observed lensing image (including residuals, chi_squareds etc.)
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
        The aspect ratio of the image, the default 'auto' scales this to the window size (see matplotlib.pyplot).
    cmap : str
        The colormap style (e.g. 'jet', 'warm', 'binary, see matplotlib.pyplot).
    title : str
        The title of the image.
    titlesize : int
        The font size of the figure title.
    xlabelsize : int
        The font size of the figure xlabel.
    ylabelsize : int
        The font size of the figure ylabel.
    output_path : str
        The path where the image is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the image is output. File formats (e.g. png, fits) output the image to harddisk. 'show' displays the image \
        in the python interpreter window.
    """

    array_plotters.plot_model_image(
        model_image=results.model_image,
        xticks=results.xticks, yticks=results.yticks, units='arcsec', xyticksize=40,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
        title='Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
        output_path=output_path, output_filename='model_image', output_format=output_format)

    array_plotters.plot_residuals(
        residuals=results.residuals,
        xticks=results.xticks, yticks=results.yticks, units='arcsec', xyticksize=40,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
        figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
        title='Residuals', titlesize=46, xlabelsize=36, ylabelsize=36,
        output_path=output_path, output_filename='residuals', output_format=output_format)

    array_plotters.plot_chi_squareds(
        chi_squareds=results.chi_squareds,
        xticks=results.xticks, yticks=results.yticks, units='arcsec', xyticksize=40,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
        figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
        title='Chi Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
        output_path=output_path, output_filename='chi_squareds', output_type=output_format)

    if isinstance(results, fitting.AbstractHyperFitter):

        contributions = results.contributions

        for i in range(len(results.hyper_galaxy_images)):

            array_plotters.plot_contributions(
                contributions=contributions[i],
                xticks=results.xticks, yticks=results.yticks, units='arcsec', xyticksize=40,
                norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
                figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                title='Contributions', titlesize=46, xlabelsize=36, ylabelsize=36,
                output_path=output_path, output_filename='contributions', output_type=output_format)

        array_plotters.plot_scaled_noise_map(
            scaled_noise_map=results.noise,
            xticks=results.xticks, yticks=results.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Scaled Noise Map', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='scaled_noise', output_format=output_format)

        array_plotters.plot_scaled_chi_squareds(
            scaled_chi_squareds=results.chi_squareds,
            xticks=results.xticks, yticks=results.yticks, units='arcsec', xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Scaled Chi Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='scaled_chi_squareds', output_format=output_format)