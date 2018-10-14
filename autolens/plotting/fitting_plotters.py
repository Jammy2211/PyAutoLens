from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import plotter_tools
from autolens.plotting import plotters
from autolens.plotting import imaging_plotters
from autolens.plotting import plane_plotters
from autolens.plotting import inversion_plotters


def plot_fitting_subplot(fit, mask=None, positions=None, units='kpc', output_path=None,
                         output_filename='fit', output_format='show', ignore_config=True):

    plot_fitting_as_subplot = conf.instance.general.get('output', 'plot_fitting_as_subplot', bool)

    if not plot_fitting_as_subplot and ignore_config is False:
        return

    if fit.tracer.total_planes == 1:

        if not fit.tracer.has_hyper_galaxy:

            plot_fitting_subplot_lens_plane_only(fit, mask, positions, units, output_path, output_filename,
                                                 output_format)

        elif fit.tracer.has_hyper_galaxy:

            plot_fitting_subplot_hyper_lens_plane_only(fit, mask, positions, units, output_path, output_filename,
                                                       output_format)

    elif fit.tracer.total_planes == 2:

        if not fit.tracer.has_hyper_galaxy:
            
            plot_fitting_subplot_lens_and_source_planes(fit, mask, positions, units, output_path, output_filename,
                                                        output_format)

def plot_fitting_subplot_lens_plane_only(fit, mask=None, positions=None, units='kpc', output_path=None,
                                         output_filename='fit', output_format='show'):
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

    rows, columns, figsize = plotter_tools.get_subplot_rows_columns_figsize(number_subplots=4)

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    imaging_plotters.plot_image(image=fit.image, mask=mask, positions=positions, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=10,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Observed Image', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 2)

    plot_model_image(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Model Image', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    plot_residuals(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Residuals', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    plot_chi_squareds(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Chi Squareds', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                       output_format=output_format)

    plt.close()

def plot_fitting_subplot_hyper_lens_plane_only(fit, mask=None, positions=None, units='kpc', output_path=None,
                                               output_filename='hyper_fit', output_format='show'):
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

    rows, columns, figsize = plotter_tools.get_subplot_rows_columns_figsize(number_subplots=9)

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    imaging_plotters.plot_image(image=fit.lensing_image.image, mask=mask, positions=positions, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=10,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Observed Image', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 2)

    plot_model_image(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Model Image', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    plot_residuals(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Residuals', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 5)

    plot_chi_squareds(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Chi Squareds', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                        output_format=output_format)

    plt.subplot(rows, columns, 4)

    plot_contributions(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Contributions', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 6)

    plot_scaled_chi_squareds(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Chi Squareds', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 8)

    imaging_plotters.plot_noise_map(image=fit.lensing_image.image, mask=mask, units=units, as_subplot=True,
                                    kpc_per_arcsec=None, xyticksize=10,
                                    norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                                    figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
                                    titlesize=10, xlabelsize=10, ylabelsize=10,
                                    output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 9)

    plot_scaled_noise_map(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Chi Squareds', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                       output_format=output_format)

    plt.close()

def plot_fitting_subplot_lens_and_source_planes(fit, mask=None, positions=None, units='kpc', output_path=None,
                                                output_filename='fit', output_format='show'):
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

    rows, columns, figsize = plotter_tools.get_subplot_rows_columns_figsize(number_subplots=6)

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    imaging_plotters.plot_image(image=fit.image, mask=mask, positions=positions, grid=None, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=10,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Observed Image', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    if fit.tracer.image_plane.has_light_profile:

        plt.subplot(rows, columns, 2)

        plot_model_image_of_plane(fit=fit, plane_index=0, as_subplot=True,
                         units=units, kpc_per_arcsec=kpc_per_arcsec,
                         xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
                         title='Lens Model Image', titlesize=10, xlabelsize=10, ylabelsize=10,
                         output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    plot_model_image_of_plane(fit=fit, plane_index=1, as_subplot=True,
                             units=units, kpc_per_arcsec=kpc_per_arcsec,
                             xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
                             title='Source Model Image', titlesize=10, xlabelsize=10, ylabelsize=10,
                             output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(2, 3, 4)

    if fit.total_inversions == 0:

        plane_plotters.plot_plane_image(plane=fit.tracer.source_plane, as_subplot=True,
            positions=None, plot_grid=False,
            units=units, kpc_per_arcsec=fit.tracer.source_plane.kpc_per_arcsec_proper,
            xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
            title='Source-plane Image', titlesize=10, xlabelsize=10, ylabelsize=10,
            output_path=output_path, output_filename='', output_format=output_format)

    # else:
    #
    #     inversion_plotters.plot_reconstruction(mapper=fit.mapper, inversion=fit.inversion,
    #         points=None, grid=None, as_subplot=True,
    #         units=units, kpc_per_arcsec=fit.kpc_per_arcsec_proper[0],
    #         xyticksize=10,
    #         norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
    #         figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
    #         title='Source-Plane Image', titlesize=10, xlabelsize=10, ylabelsize=10,
    #         output_path=output_path, output_filename=None, output_format=output_format)


    plt.subplot(rows, columns, 5)

    plot_residuals(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Residuals', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 6)

    plot_chi_squareds(fit=fit, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec,
        xyticksize=10, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
        title='Chi Squareds', titlesize=10, xlabelsize=10, ylabelsize=10,
        output_path=output_path, output_filename='', output_format=output_format)

    plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                       output_format=output_format)

    plt.close()


def plot_fitting_individuals(fit, mask=None, positions=None, units='kpc', output_path=None, output_format='show'):

    if fit.tracer.total_planes == 1:

        if not fit.tracer.has_hyper_galaxy:

            plot_fitting_individuals_lens_plane_only(fit, mask, positions, units, output_path, output_format)


        elif fit.tracer.has_hyper_galaxy:
       
                    plot_fitting_individuals_hyper_lens_plane_only(fit, mask, positions, units, output_path, 
                                                                   output_format)
        
    elif fit.tracer.total_planes == 2:

        if not fit.tracer.has_hyper_galaxy:

            plot_fitting_individuals_lens_and_source_planes(fit, mask, positions, units, output_path, output_format)

def plot_fitting_individuals_lens_plane_only(fit, mask=None, positions=None, units='kpc', output_path=None, 
                                             output_format='show'):
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

    plot_fitting_model_image = conf.instance.general.get('output', 'plot_fitting_model_image', bool)
    plot_fitting_residuals = conf.instance.general.get('output', 'plot_fitting_residuals', bool)
    plot_fitting_chi_squareds = conf.instance.general.get('output', 'plot_fitting_chi_squareds', bool)

    if plot_fitting_model_image:

        plot_model_image(fit=fit, output_path=output_path, output_format=output_format)

    if plot_fitting_residuals:

        plot_residuals(fit=fit, output_path=output_path, output_format=output_format)

    if plot_fitting_chi_squareds:

        plot_chi_squareds(fit=fit, output_path=output_path, output_format=output_format)


def plot_fitting_individuals_hyper_lens_plane_only(fit, mask=None, positions=None,  units='kpc', output_path=None, 
                                                   output_format='show'):
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

    plot_fitting_model_image = conf.instance.general.get('output', 'plot_fitting_model_image', bool)
    plot_fitting_residuals = conf.instance.general.get('output', 'plot_fitting_residuals', bool)
    plot_fitting_chi_squareds = conf.instance.general.get('output', 'plot_fitting_chi_squareds', bool)
    plot_fitting_contributions = conf.instance.general.get('output', 'plot_fitting_contributions', bool)
    plot_fitting_scaled_chi_squareds = conf.instance.general.get('output', 'plot_fitting_scaled_chi_squareds', bool)
    plot_fitting_scaled_noise_map = conf.instance.general.get('output', 'plot_fitting_scaled_noise_map', bool)

    if plot_fitting_model_image:

        plot_model_image(fit=fit, output_path=output_path, output_format=output_format)

    if plot_fitting_residuals:

        plot_residuals(fit=fit, output_path=output_path, output_format=output_format)

    if plot_fitting_chi_squareds:

        plot_chi_squareds(fit=fit, output_path=output_path, output_format=output_format)

    if plot_fitting_contributions:

        plot_contributions(fit=fit, output_path=output_path, output_format=output_format)

    if plot_fitting_scaled_noise_map:

        plot_scaled_noise_map(fit=fit, output_path=output_path, output_format=output_format)

    if plot_fitting_scaled_chi_squareds:

        plot_scaled_chi_squareds(fit=fit, output_path=output_path, output_format=output_format)

def plot_fitting_individuals_lens_and_source_planes(fit, mask=None, positions=None, units='kpc', output_path=None, 
                                                    output_format='show'):
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

    plot_fitting_model_image = conf.instance.general.get('output', 'plot_fitting_model_image', bool)
    plot_fitting_lens_model_image = conf.instance.general.get('output', 'plot_fitting_lens_model_image', bool)
    plot_fitting_source_model_image = conf.instance.general.get('output', 'plot_fitting_source_model_image', bool)
    plot_fitting_source_plane_image = conf.instance.general.get('output', 'plot_fitting_source_plane_image', bool)
    plot_fitting_residuals = conf.instance.general.get('output', 'plot_fitting_residuals', bool)
    plot_fitting_chi_squareds = conf.instance.general.get('output', 'plot_fitting_chi_squareds', bool)

    if plot_fitting_model_image:

        plot_model_image(fit=fit, output_path=output_path, output_format=output_format)

    if plot_fitting_lens_model_image:

        plot_model_image_of_plane(fit=fit, plane_index=0, output_path=output_path,
                                  output_filename='fit_lens_plane_model_image', output_format=output_format)

    if plot_fitting_source_model_image:

        plot_model_image_of_plane(fit=fit, plane_index=1, output_path=output_path,
                                  output_filename='fit_source_plane_model_image', output_format=output_format)

    if plot_fitting_source_plane_image:

        plane_plotters.plot_plane_image(plane=fit.tracer.source_plane, positions=None, plot_grid=False,
            output_path=output_path, output_filename='fit_source_plane', output_format=output_format)

    if plot_fitting_residuals:

        plot_residuals(fit=fit, output_path=output_path, output_format=output_format)

    if plot_fitting_chi_squareds:

        plot_chi_squareds(fit=fit, output_path=output_path, output_format=output_format)

def plot_model_image(fit, as_subplot=False, 
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Fit Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='fit_model_image'):

    plotters.plot_model_image(model_image=fit.model_image, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize, 
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize, 
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_model_image_of_plane(fit, plane_index, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Fit Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='fit_model_image_of_plane'):

    plotters.plot_model_image(model_image=fit.model_images_of_planes[plane_index], as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_residuals(fit, as_subplot=False, 
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Fit Residuals', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='fit_residuals'):

    plotters.plot_residuals(residuals=fit.residuals, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_chi_squareds(fit, as_subplot=False, 
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Fit Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='fit_chi_squareds'):

    plotters.plot_chi_squareds(chi_squareds=fit.chi_squareds, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_contributions(fit, as_subplot=False, 
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Contributions', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='fit_contributions'):

    if len(fit.contributions) > 1:
        contributions = sum(fit.contributions)
    else:
        contributions = fit.contributions[0]

    plotters.plot_contributions(contributions=contributions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_scaled_model_image(fit, as_subplot=False, 
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Fit Scaled Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='fit_scaled_model_image'):

    plotters.plot_scaled_model_image(scaled_model_image=fit.scaled_model_image, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_scaled_residuals(fit, as_subplot=False, 
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Fit Scaled Residuals', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='fit_scaled_residuals'):

    plotters.plot_scaled_residuals(scaled_residuals=fit.scaled_residuals, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_scaled_chi_squareds(fit, as_subplot=False, 
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Fit Scaled Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='fit_scaled_chi_squareds'):

    plotters.plot_scaled_chi_squareds(scaled_chi_squareds=fit.scaled_chi_squareds, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_scaled_noise_map(fit, as_subplot=False, 
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Fit Scaled Noise Map', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='fit_scaled_noise_map'):

    plotters.plot_scaled_noise_map(scaled_noise_map=fit.scaled_noise_map, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

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
