from autolens.visualize import array_plotters

def plot_image_data_from_image(image, units, xyticksize=40,
                               norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                               figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                               image_title='Obsserved Image', noise_map_title='Noise Map', psf_title='PSF',
                               titlesize=46, xlabelsize=36, ylabelsize=36,
                               output_path=None,
                               output_image_filename='observed_image', output_noise_map_filename='noise_map',
                               output_psf_filename='psf', output_format='show'):
    """Plot the observed image of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    image : autolens.imaging.image.Image
        Class containing the image, noise-map and PSF that are to be plotted.
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
        The title of the noise-map image.
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

    array_plotters.plot_observed_image_array(
        array=image, units=units, xticks=image.xticks, yticks=image.yticks, xyticksize=xyticksize,
        norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
        title=image_title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        output_path=output_path, output_filename=output_image_filename, output_format=output_format)

    array_plotters.plot_noise_map_array(
        array=image.noise_map, units=units, xticks=image.xticks, yticks=image.yticks, xyticksize=xyticksize,
        norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
        title=noise_map_title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        output_path=output_path, output_filename=output_noise_map_filename, output_format=output_format)

    array_plotters.plot_psf_array(
        array=image.psf, units=units, xticks=image.xticks, yticks=image.yticks, xyticksize=xyticksize,
        norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
        title=psf_title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        output_path=output_path, output_filename=output_psf_filename, output_format=output_format)

def plot_model_image_from_fitter(fitter, units, xyticksize=40,
                                 norm='symmetric_log', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                                 figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                 title='Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
                                 output_path=None, output_filename='model_image', output_format='show'):
    """Plot the model image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fitter : autolens.lensing.fitting.Fitter
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
    array_plotters.plot_model_image_array(
        array=fitter.model_image, units=units,
        xticks=fitter.lensing_image.image.xticks, yticks=fitter.lensing_image.image.yticks, xyticksize=xyticksize,
        norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_residuals_from_fitter(fitter, units, xyticksize=40,
                               norm='symmetric_log', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
                               figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                               title='Residuals', titlesize=46, xlabelsize=36, ylabelsize=36,
                               output_path=None, output_filename='residuals', output_format='show'):
    """Plot the residuals of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fitter : autolens.lensing.fitting.Fitter
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
    array_plotters.plot_residuals_array(
        array=fitter.residuals, units=units,
        xticks=fitter.lensing_image.image.xticks, yticks=fitter.lensing_image.image.yticks, xyticksize=xyticksize,
        norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_chi_squareds_from_fitter(fitter, units, xyticksize=40,
                                  norm='log', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
                                  figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                  title='Chi Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
                                  output_path=None, output_filename='chi_squareds', output_format='show'):
    """Plot the chi squareds of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fitter : autolens.lensing.fitting.Fitter
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
    array_plotters.plot_chi_squareds_array(
        array=fitter.chi_squareds, units=units,
        xticks=fitter.lensing_image.image.xticks, yticks=fitter.lensing_image.image.yticks, xyticksize=xyticksize,
        norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        output_path=output_path, output_filename=output_filename, output_type=output_format)

def plot_scaled_noise_map_from_fitter(fitter, units, xyticksize=40,
                                  norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
                                  figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                  title='Scaled Noise-Map', titlesize=46, xlabelsize=36, ylabelsize=36,
                                  output_path=None, output_filename='scaled_noise', output_format='show'):
    """Plot the chi squareds of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fitter : autolens.lensing.fitting.Fitter
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
    array_plotters.plot_scaled_noise_map_array(
        array=fitter.scaled_noise, units=units,
        xticks=fitter.lensing_image.image.xticks, yticks=fitter.lensing_image.image.yticks, xyticksize=xyticksize,
        norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        output_path=output_path, output_filename=output_filename, output_type=output_format)

def plot_scaled_chi_squareds_from_fitter(fitter, units, xyticksize=40,
                                  norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
                                  figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                  title='Scaled Chi Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
                                  output_path=None, output_filename='scaled_chi_squareds', output_format='show'):
    """Plot the chi squareds of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fitter : autolens.lensing.fitting.Fitter
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
    array_plotters.plot_scaled_chi_squareds_array(
        array=fitter.scaled_chi_squareds, units=units,
        xticks=fitter.lensing_image.image.xticks, yticks=fitter.lensing_image.image.yticks, xyticksize=xyticksize,
        norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        output_path=output_path, output_filename=output_filename, output_type=output_format)