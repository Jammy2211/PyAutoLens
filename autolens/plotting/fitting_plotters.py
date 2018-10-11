from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import array_plotters
from autolens.plotting import inversion_plotters


def plot_fitting(fit, units='kpc', output_path=None, output_filename='fit', output_format='show', ignore_config=True):
    plot_fitting_as_subplot = conf.instance.general.get('output', 'plot_fitting_as_subplot', bool)

    if not plot_fitting_as_subplot and ignore_config is False:
        return

    if fit.total_planes == 1:

        if not fit.is_hyper_fit:

            plot_fitting_lens_plane_only(fit, units, output_path, output_filename, output_format)

        elif fit.is_hyper_fit:

            plot_fitting_hyper_lens_plane_only(fit, units, output_path, output_filename, output_format)

    elif fit.total_planes == 2:

        if not fit.is_hyper_fit:
            plot_fitting_lens_and_source_planes(fit, units, output_path, output_filename, output_format)


def plot_fitting_lens_plane_only(fit, units='kpc', output_path=None, output_filename='fit', output_format='show'):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
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
        array=fit.image, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.image.xticks, yticks=fit.image.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Observed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 2, 2)

    array_plotters.plot_array(
        array=fit.model_image, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.model_image.xticks, yticks=fit.model_image.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 2, 3)

    array_plotters.plot_array(
        array=fit.residuals, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.residuals.xticks, yticks=fit.residuals.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Residuals', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 2, 4)

    array_plotters.plot_array(
        array=fit.chi_squareds, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.chi_squareds.xticks, yticks=fit.chi_squareds.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Chi Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    array_plotters.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                        output_format=output_format)
    plt.close()


def plot_fitting_hyper_lens_plane_only(fit, units='kpc', output_path=None, output_filename='fit', output_format='show'):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
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
        array=fit.image, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.image.xticks, yticks=fit.image.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Observed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 2)

    array_plotters.plot_array(
        array=fit.model_image, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.model_image.xticks, yticks=fit.model_image.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 3)

    array_plotters.plot_array(
        array=fit.residuals, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.residuals.xticks, yticks=fit.residuals.yticks, xyticksize=16,
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
        array=contributions, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=contributions.xticks, yticks=contributions.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Lens-Plane Contributions', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 5)

    array_plotters.plot_array(
        array=fit.chi_squareds, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.chi_squareds.xticks, yticks=fit.chi_squareds.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 6)

    array_plotters.plot_array(
        array=fit.scaled_chi_squareds, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.scaled_chi_squareds.xticks, yticks=fit.scaled_chi_squareds.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Scaled Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 8)

    array_plotters.plot_array(
        array=fit.noise_map, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.noise_map.xticks, yticks=fit.noise_map.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Noise Map', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(3, 3, 9)

    array_plotters.plot_array(
        array=fit.scaled_noise_map, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.scaled_noise_map.xticks, yticks=fit.scaled_noise_map.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Scaled Noise Map', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    array_plotters.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()


def plot_fitting_lens_and_source_planes(fit, units='kpc', output_path=None, output_filename='fit', output_format='show'):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 3, 1)

    array_plotters.plot_array(
        array=fit.image, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.image.xticks, yticks=fit.image.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Observed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    if fit.model_images_of_planes[0] is not None:

        plt.subplot(2, 3, 2)

        array_plotters.plot_array(
            array=fit.model_images_of_planes[0], points=None, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.image.xticks, yticks=fit.image.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Lens Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 3, 3)

    array_plotters.plot_array(
        array=fit.model_images_of_planes[1], points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.model_images_of_planes[1].xticks, yticks=fit.model_images_of_planes[1].yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Source Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 3, 4)

    if fit.total_inversions == 0:

        array_plotters.plot_array(
            array=fit.plane_images[1], grid=fit.plane_images[1].grid, points=None, as_subplot=True,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[1],
            xticks=fit.plane_images[1].xticks, yticks=fit.plane_images[1].yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Source-Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

    else:

        inversion_plotters.plot_reconstruction(mapper=fit.mapper, inversion=fit.inversion,
            points=None, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Source-Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)


    plt.subplot(2, 3, 5)

    array_plotters.plot_array(
        array=fit.residuals, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.residuals.xticks, yticks=fit.residuals.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Residuals', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(2, 3, 6)

    array_plotters.plot_array(
        array=fit.chi_squareds, points=None, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
        xticks=fit.chi_squareds.xticks, yticks=fit.chi_squareds.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    array_plotters.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()


def plot_fitting_individuals(fit, units='kpc', plot_model_image=False, plot_lens_model_image=False, plot_source_model_image=False,
                             plot_source_plane_image=False, plot_residuals=False, plot_chi_squareds=False,
                             plot_contributions=False, plot_scaled_chi_squareds=False, plot_scaled_noise_map=False,
                             output_path=None,  output_format='show', ignore_config=True):

    if fit.total_planes == 1:

        if not fit.is_hyper_fit:

            plot_fitting_individuals_lens_plane_only(fit, plot_model_image, plot_residuals, plot_chi_squareds,
                                                     units, output_path, output_format, ignore_config)

        elif fit.is_hyper_fit:

            plot_fitting_individuals_hyper_lens_plane_only(fit, plot_model_image, plot_residuals, plot_chi_squareds,
                                                           plot_contributions, plot_scaled_chi_squareds,
                                                           plot_scaled_noise_map, units, output_path, output_format,
                                                           ignore_config)

    elif fit.total_planes == 2:

        if not fit.is_hyper_fit:

            plot_fitting_individuals_lens_and_source_planes(fit, plot_model_image, plot_lens_model_image,
                                                            plot_source_model_image, plot_source_plane_image,
                                                            plot_residuals, plot_chi_squareds, units, output_path,
                                                            output_format, ignore_config)


def plot_fitting_individuals_lens_plane_only(fit, plot_model_image, plot_residuals, plot_chi_squareds,
                                             units='kpc', output_path=None, output_format='show', ignore_config=True):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    if not ignore_config:

        plot_model_image = conf.instance.general.get('output', 'plot_fitting_model_image', bool)
        plot_residuals = conf.instance.general.get('output', 'plot_fitting_residuals', bool)
        plot_chi_squareds = conf.instance.general.get('output', 'plot_fitting_chi_squareds', bool)

    if plot_model_image:
        array_plotters.plot_array(
            array=fit.model_image, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.model_image.xticks, yticks=fit.model_image.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='model_image', output_format=output_format)

    if plot_residuals:

        array_plotters.plot_array(
            array=fit.residuals, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.residuals.xticks, yticks=fit.residuals.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Residuals', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='residuals', output_format=output_format)

    if plot_chi_squareds:

        array_plotters.plot_array(
            array=fit.chi_squareds, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.chi_squareds.xticks, yticks=fit.chi_squareds.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Chi-Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='chi_squareds', output_format=output_format)


def plot_fitting_individuals_hyper_lens_plane_only(fit, plot_model_image, plot_residuals, plot_chi_squareds,
                                                   plot_contributions, plot_scaled_chi_squareds, plot_scaled_noise_map,
                                                   units='kpc', output_path=None, output_format='show',
                                                   ignore_config=True):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    if not ignore_config:

        plot_model_image = conf.instance.general.get('output', 'plot_fitting_model_image', bool)
        plot_residuals = conf.instance.general.get('output', 'plot_fitting_residuals', bool)
        plot_chi_squareds = conf.instance.general.get('output', 'plot_fitting_chi_squareds', bool)
        plot_contributions = conf.instance.general.get('output', 'plot_fitting_contributions', bool)
        plot_scaled_chi_squareds = conf.instance.general.get('output', 'plot_fitting_scaled_chi_squareds', bool)
        plot_scaled_noise_map = conf.instance.general.get('output', 'plot_fitting_scaled_noise_map', bool)

    if plot_model_image:
        array_plotters.plot_array(
            array=fit.model_image, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.model_image.xticks, yticks=fit.model_image.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Observed Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='model_image', output_format=output_format)

    if plot_residuals:
        array_plotters.plot_array(
            array=fit.residuals, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.residuals.xticks, yticks=fit.residuals.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Residuals', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='residuals', output_format=output_format)

    if plot_chi_squareds:
        array_plotters.plot_array(
            array=fit.chi_squareds, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.chi_squareds.xticks, yticks=fit.chi_squareds.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Chi Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='chi_squareds', output_format=output_format)

    if plot_contributions:

        if len(fit.contributions) > 1:
            contributions = sum(fit.contributions)
        else:
            contributions = fit.contributions[0]

        array_plotters.plot_array(
            array=contributions, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=contributions.xticks, yticks=contributions.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Lens-Plane Contributions', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='lens_plane_contributions', output_format=output_format)

    if plot_scaled_noise_map:
        array_plotters.plot_array(
            array=fit.scaled_noise_map, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.scaled_noise_map.xticks, yticks=fit.scaled_noise_map.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Scaled Noise Map', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='scaled_noise_map', output_format=output_format)

    if plot_scaled_chi_squareds:
        array_plotters.plot_array(
            array=fit.scaled_chi_squareds, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.scaled_chi_squareds.xticks, yticks=fit.scaled_chi_squareds.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Scaled Chi-Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path,output_filename='scaled_chi_squareds', output_format=output_format)


def plot_fitting_individuals_lens_and_source_planes(fit, plot_model_image, plot_lens_model_image,
                                                    plot_source_model_image, plot_source_plane_image, plot_residuals,
                                                    plot_chi_squareds,
                                                    units='kpc', output_path=None, output_format='show',
                                                    ignore_config=True):
    """Plot the model _image of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model _image and observed lensing _image (including residuals, chi_squareds etc.)
    output_path : str
        The path where the _image is output if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the _image is output. File formats (e.g. png, fittings) output the _image to harddisk. 'show' displays the _image \
        in the python interpreter window.
    """

    if not ignore_config:

        plot_model_image = conf.instance.general.get('output', 'plot_fitting_model_image', bool)
        plot_lens_model_image = conf.instance.general.get('output', 'plot_fitting_lens_model_image', bool)
        plot_source_model_image = conf.instance.general.get('output', 'plot_fitting_source_model_image', bool)
        plot_source_plane_image = conf.instance.general.get('output', 'plot_fitting_source_plane_image', bool)
        plot_residuals = conf.instance.general.get('output', 'plot_fitting_residuals', bool)
        plot_chi_squareds = conf.instance.general.get('output', 'plot_fitting_chi_squareds', bool)

    if plot_model_image:
        array_plotters.plot_array(
            array=fit.model_image, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.model_image.xticks, yticks=fit.model_image.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='model_image', output_format=output_format)

    if plot_lens_model_image:

        array_plotters.plot_array(
            array=fit.model_images_of_planes[0], points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.image.xticks, yticks=fit.image.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Lens Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='lens_model_image', output_format=output_format)

    if plot_source_model_image:

        array_plotters.plot_array(
            array=fit.model_images_of_planes[1], points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.model_images_of_planes[1].xticks, yticks=fit.model_images_of_planes[1].yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Source Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='source_model_image', output_format=output_format)

    if plot_source_plane_image:
        array_plotters.plot_array(
            array=fit.plane_images[1], points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[1],
            xticks=fit.plane_images[1].xticks, yticks=fit.plane_images[1].yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Source-Plane Image', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='source_plane', output_format=output_format)

    if plot_residuals:
        array_plotters.plot_array(
            array=fit.residuals, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.residuals.xticks, yticks=fit.residuals.yticks, xyticksize=40,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
            title='Residuals', titlesize=46, xlabelsize=36, ylabelsize=36,
            output_path=output_path, output_filename='residuals', output_format=output_format)

    if plot_chi_squareds:
        array_plotters.plot_array(
            array=fit.chi_squareds, points=None, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
            xticks=fit.residuals.xticks, yticks=fit.residuals.yticks, xyticksize=40,
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
