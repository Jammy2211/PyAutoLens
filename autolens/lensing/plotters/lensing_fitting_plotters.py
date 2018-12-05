from matplotlib import pyplot as plt

from autofit import conf
from autolens.data.array.plotters import plotter_util
from autolens.data.imaging.plotters import imaging_plotters
from autolens.lensing.plotters import plane_plotters
from autolens.model.inversion.plotters import inversion_plotters
from autolens.data.fitting.plotters import fitting_plotters


def plot_fitting_subplot(fit, should_plot_mask=True, positions=None, should_plot_image_plane_pix=True,
                         units='arcsec', figsize=None, aspect='equal',
                         cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                         titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                         mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                         output_path=None, output_filename='lensing_fit', output_format='show', ignore_config=True):
    plot_lensing_fitting_as_subplot = conf.instance.general.get('output', 'plot_lensing_fitting_as_subplot', bool)

    if not plot_lensing_fitting_as_subplot and ignore_config is False:
        return

    if fit.tracer.total_planes == 1:

        if not fit.tracer.has_hyper_galaxy:

            plot_fitting_subplot_lens_plane_only(fit=fit, should_plot_mask=should_plot_mask, positions=positions,
                                                 should_plot_image_plane_pix=should_plot_image_plane_pix,
                                                 units=units, figsize=figsize,
                                                 aspect=aspect,
                                                 cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                                                 linthresh=linthresh,
                                                 linscale=linscale,
                                                 cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                                 titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                                 xyticksize=xyticksize,
                                                 mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                                                 grid_pointsize=grid_pointsize,
                                                 output_path=output_path, output_filename=output_filename,
                                                 output_format=output_format)
        elif fit.tracer.has_hyper_galaxy:

            plot_fitting_subplot_hyper_lens_plane_only(fit=fit, should_plot_mask=should_plot_mask, positions=positions,
                                                       should_plot_image_plane_pix=should_plot_image_plane_pix,
                                                       units=units, figsize=figsize,
                                                       aspect=aspect,
                                                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                                                       linthresh=linthresh,
                                                       linscale=linscale,
                                                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                                       titlesize=titlesize, xlabelsize=xlabelsize,
                                                       ylabelsize=ylabelsize,
                                                       xyticksize=xyticksize,
                                                       mask_pointsize=mask_pointsize,
                                                       position_pointsize=position_pointsize,
                                                       grid_pointsize=grid_pointsize,
                                                       output_path=output_path, output_filename=output_filename,
                                                       output_format=output_format)

    elif fit.tracer.total_planes == 2:

        if not fit.tracer.has_hyper_galaxy:
            plot_fitting_subplot_lens_and_source_planes(fit=fit, should_plot_mask=should_plot_mask, positions=positions,
                                                        should_plot_image_plane_pix=should_plot_image_plane_pix,
                                                        units=units, figsize=figsize,
                                                        aspect=aspect,
                                                        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                                                        linthresh=linthresh,
                                                        linscale=linscale,
                                                        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                                        titlesize=titlesize, xlabelsize=xlabelsize,
                                                        ylabelsize=ylabelsize,
                                                        xyticksize=xyticksize,
                                                        mask_pointsize=mask_pointsize,
                                                        position_pointsize=position_pointsize,
                                                        grid_pointsize=grid_pointsize,
                                                        output_path=output_path, output_filename=output_filename,
                                                        output_format=output_format)


def plot_fitting_subplot_lens_plane_only(fit, should_plot_mask=True, positions=None, should_plot_image_plane_pix=True,
                                         units='arcsec', figsize=None, aspect='equal',
                                         cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                         linscale=0.01,
                                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                         titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                                         mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                                         output_path=None, output_filename='lensing_fit', output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model datas_ and observed lensing datas_ (including residuals, chi_squareds etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fittings) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=4)

    mask = fitting_plotters.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    image_plane_pix_grid = get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

    imaging_plotters.plot_image(image=fit.images[0], mask=mask, positions=positions,
                                image_plane_pix_grid=image_plane_pix_grid,
                                as_subplot=True,
                                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                linscale=linscale,
                                cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                xyticksize=xyticksize,
                                grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
                                mask_pointsize=mask_pointsize,
                                output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 2)

    fitting_plotters.plot_model_image(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                      units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                      cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                      linscale=linscale,
                                      cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                      titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                      output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    fitting_plotters.plot_residuals(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                    cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                    output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    fitting_plotters.plot_chi_squareds(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                       units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                       linscale=linscale,
                                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                       titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                       output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()


def plot_fitting_subplot_hyper_lens_plane_only(fit, should_plot_mask=True, positions=None,
                                               should_plot_image_plane_pix=True,
                                               units='arcsec', figsize=None, aspect='equal',
                                               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                               linscale=0.01,
                                               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                               titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                                               mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                                               output_path=None, output_filename='lensing_hyper_fit', output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model datas_ and observed lensing datas_ (including residuals, chi_squareds etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fittings) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=9)

    mask = fitting_plotters.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    image_plane_pix_grid = get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

    imaging_plotters.plot_image(image=fit.images[0], mask=mask, positions=positions,
                                image_plane_pix_grid=image_plane_pix_grid, as_subplot=True,
                                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                linscale=linscale,
                                cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                xyticksize=xyticksize,
                                grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
                                mask_pointsize=mask_pointsize,
                                output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 2)

    fitting_plotters.plot_model_image(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                      units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                      cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
                                      cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                      titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                      output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    fitting_plotters.plot_residuals(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                    cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                    output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 5)

    fitting_plotters.plot_chi_squareds(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                       units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                       linscale=linscale,
                                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                       titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                       output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.subplot(rows, columns, 4)

    fitting_plotters.plot_contributions(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                        linscale=linscale,
                                        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 6)

    fitting_plotters.plot_scaled_chi_squareds(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                              linscale=linscale,
                                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                              titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                              output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 8)

    imaging_plotters.plot_noise_map(image=fit.images[0], mask=mask, as_subplot=True,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                    cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                    xyticksize=xyticksize,
                                    output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 9)

    fitting_plotters.plot_scaled_noise_map(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                           linscale=linscale,
                                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                           titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                           output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()


def plot_fitting_subplot_lens_and_source_planes(fit, should_plot_mask=True, should_plot_source_grid=False, positions=None,
                                                should_plot_image_plane_pix=True,
                                                units='arcsec', figsize=None, aspect='equal',
                                                cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                                linscale=0.01,
                                                cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                                titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                                                mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                                                output_path=None, output_filename='lensing_fit', output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model datas_ and observed lensing datas_ (including residuals, chi_squareds etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fittings) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=6)

    mask = fitting_plotters.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    image_plane_pix_grid = get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

    imaging_plotters.plot_image(image=fit.images[0], mask=mask, positions=positions,
                                image_plane_pix_grid=image_plane_pix_grid, as_subplot=True,
                                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                linscale=linscale,
                                cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                xyticksize=xyticksize,
                                grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
                                mask_pointsize=mask_pointsize,
                                output_path=output_path, output_filename='', output_format=output_format)

    if fit.tracer.image_plane.has_light_profile:

        plt.subplot(rows, columns, 2)

        fitting_plotters.plot_model_image_of_plane(fit=fit, image_index=0, plane_index=0, mask=mask, as_subplot=True,
                                                   units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                                   cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                                   linscale=linscale,
                                                   cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                                   titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                                   xyticksize=xyticksize,
                                                   output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    fitting_plotters.plot_model_image_of_plane(fit=fit, image_index=0, plane_index=1, mask=mask, as_subplot=True,
                                               units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                               cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                               linscale=linscale,
                                               cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                               titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                               output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    if fit.total_inversions == 0:

        plane_plotters.plot_plane_image(plane=fit.tracer.source_plane, image_index=0, as_subplot=True,
                                        positions=None, plot_grid=should_plot_source_grid,
                                        units=units, figsize=figsize, aspect=aspect,
                                        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                        linscale=linscale,
                                        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                        xyticksize=xyticksize,
                                        grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
                                        output_path=output_path, output_filename='', output_format=output_format)

    elif fit.total_inversions == 1:

        inversion_plotters.plot_reconstructed_pixelization(inversion=fit.inversion, positions=None,
                                                           should_plot_grid=False, should_plot_centres=False,
                                                           as_subplot=True,
                                                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize,
                                                           aspect=aspect,
                                                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                                                           linthresh=linthresh, linscale=linscale,
                                                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction,
                                                           cb_pad=cb_pad,
                                                           titlesize=titlesize, xlabelsize=xlabelsize,
                                                           ylabelsize=ylabelsize, xyticksize=xyticksize,
                                                           output_path=output_path, output_filename=None,
                                                           output_format=output_format)

    plt.subplot(rows, columns, 5)

    fitting_plotters.plot_residuals(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                    cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                    output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 6)

    fitting_plotters.plot_chi_squareds(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                       units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                       linscale=linscale,
                                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                       titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                       output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()


def plot_fitting_individuals(fit, units='kpc', output_path=None, output_format='show'):
    if fit.tracer.total_planes == 1:

        if not fit.tracer.has_hyper_galaxy:

            plot_fitting_individuals_lens_plane_only(fit, units, output_path, output_format)


        elif fit.tracer.has_hyper_galaxy:

            plot_fitting_individuals_hyper_lens_plane_only(fit, units, output_path, output_format)

    elif fit.tracer.total_planes == 2:

        if not fit.tracer.has_hyper_galaxy:
            plot_fitting_individuals_lens_and_source_planes(fit, units, output_path, output_format)


def plot_fitting_individuals_lens_plane_only(fit, units='kpc', output_path=None, output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model datas_ and observed lensing datas_ (including residuals, chi_squareds etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fittings) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    plot_lensing_fitting_model_image = conf.instance.general.get('output', 'plot_lensing_fitting_model_image', bool)
    plot_lensing_fitting_residuals = conf.instance.general.get('output', 'plot_lensing_fitting_residuals', bool)
    plot_lensing_fitting_chi_squareds = conf.instance.general.get('output', 'plot_lensing_fitting_chi_squareds', bool)

    mask = fitting_plotters.get_mask(fit=fit, should_plot_mask=True)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    if plot_lensing_fitting_model_image:
        fitting_plotters.plot_model_image(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                          output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_residuals:
        fitting_plotters.plot_residuals(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                        output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_chi_squareds:
        fitting_plotters.plot_chi_squareds(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                           output_path=output_path, output_format=output_format)


def plot_fitting_individuals_hyper_lens_plane_only(fit, units='kpc', output_path=None, output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model datas_ and observed lensing datas_ (including residuals, chi_squareds etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fittings) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    plot_lensing_fitting_model_image = \
        conf.instance.general.get('output', 'plot_lensing_fitting_model_image', bool)
    plot_lensing_fitting_residuals = \
        conf.instance.general.get('output', 'plot_lensing_fitting_residuals', bool)
    plot_lensing_fitting_chi_squareds = \
        conf.instance.general.get('output', 'plot_lensing_fitting_chi_squareds', bool)
    plot_lensing_fitting_contributions = \
        conf.instance.general.get('output', 'plot_lensing_fitting_contributions', bool)
    plot_lensing_fitting_scaled_chi_squareds = \
        conf.instance.general.get('output', 'plot_lensing_fitting_scaled_chi_squareds', bool)
    plot_lensing_fitting_scaled_noise_map = \
        conf.instance.general.get('output', 'plot_lensing_fitting_scaled_noise_map', bool)

    mask = fitting_plotters.get_mask(fit=fit, should_plot_mask=True)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    if plot_lensing_fitting_model_image:
        fitting_plotters.plot_model_image(fit=fit, image_index=0, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                          output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_residuals:
        fitting_plotters.plot_residuals(fit=fit, image_index=0, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                        output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_chi_squareds:
        fitting_plotters.plot_chi_squareds(fit=fit, image_index=0, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                           output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_contributions:
        fitting_plotters.plot_contributions(fit=fit, image_index=0, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                            output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_scaled_noise_map:
        fitting_plotters.plot_scaled_noise_map(fit=fit, image_index=0, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                               output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_scaled_chi_squareds:
        fitting_plotters.plot_scaled_chi_squareds(fit=fit, image_index=0, mask=mask, units=units,
                                                  kpc_per_arcsec=kpc_per_arcsec, output_path=output_path,
                                                  output_format=output_format)


def plot_fitting_individuals_lens_and_source_planes(fit, units='kpc', output_path=None, output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model datas_ and observed lensing datas_ (including residuals, chi_squareds etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fittings) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    plot_lensing_fitting_model_image = \
        conf.instance.general.get('output', 'plot_lensing_fitting_model_image', bool)
    plot_lensing_fitting_lens_model_image = \
        conf.instance.general.get('output', 'plot_lensing_fitting_lens_model_image', bool)
    plot_lensing_fitting_source_model_image = \
        conf.instance.general.get('output', 'plot_lensing_fitting_source_model_image', bool)
    plot_lensing_fitting_source_plane_image = \
        conf.instance.general.get('output', 'plot_lensing_fitting_source_plane_image', bool)
    plot_lensing_fitting_residuals = conf.instance.general.get('output', 'plot_lensing_fitting_residuals', bool)
    plot_lensing_fitting_chi_squareds = conf.instance.general.get('output', 'plot_lensing_fitting_chi_squareds', bool)

    mask = fitting_plotters.get_mask(fit=fit, should_plot_mask=True)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    if plot_lensing_fitting_model_image:
        fitting_plotters.plot_model_image(fit=fit, image_index=0, units=units, mask=mask, kpc_per_arcsec=kpc_per_arcsec,
                                          output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_lens_model_image:
        fitting_plotters.plot_model_image_of_plane(fit=fit, image_index=0, plane_index=0, mask=mask, units=units,
                                                   kpc_per_arcsec=kpc_per_arcsec,
                                                   output_path=output_path, output_filename='fit_lens_plane_model_image',
                                                   output_format=output_format)

    if plot_lensing_fitting_source_model_image:
        fitting_plotters.plot_model_image_of_plane(fit=fit, image_index=0, plane_index=1, mask=mask, units=units,
                                                   kpc_per_arcsec=kpc_per_arcsec,
                                                   output_path=output_path, output_filename='fit_source_plane_model_image',
                                                   output_format=output_format)

    if plot_lensing_fitting_source_plane_image:

        if fit.total_inversions == 0:

           plane_plotters.plot_plane_image(plane=fit.tracer.source_plane, image_index=0, plot_grid=True,
                                           units=units, figsize=(20, 20),
                                           output_path=output_path, output_filename='fit_source_plane',
                                           output_format=output_format)

        elif fit.total_inversions == 1:

            inversion_plotters.plot_reconstructed_pixelization(inversion=fit.inversion, should_plot_grid=True,
                                                               units=units, figsize=(20, 20),
                                                               output_path=output_path, output_filename='fit_source_plane',
                                                               output_format=output_format)

    if plot_lensing_fitting_residuals:
        fitting_plotters.plot_residuals(fit=fit, image_index=0, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                        output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_chi_squareds:
        fitting_plotters.plot_chi_squareds(fit=fit, image_index=0, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                           output_path=output_path, output_format=output_format)

def get_image_plane_pix_grid(should_plot_image_plane_pix, fit):
    if hasattr(fit, 'mapper'):
        if should_plot_image_plane_pix and fit.mapper.is_image_plane_pixelization:
            return fit.tracer.image_plane.grids[0].pix
    else:
        return None