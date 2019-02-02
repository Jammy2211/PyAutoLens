from matplotlib import pyplot as plt

from autofit import conf
from autolens.data.array.plotters import plotter_util
from autolens.lens.plotters import lens_plotter_util
from autolens.lens.plotters import plane_plotters
from autolens.model.inversion.plotters import inversion_plotters


def plot_fit_subplot(
        fit, should_plot_mask=True, extract_mask_region=False, positions=None, should_plot_image_plane_pix=True,
        units='arcsec', figsize=None, aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10, grid_pointsize=1,
        output_path=None, output_filename='lens_fit', output_format='show', ignore_config=True):

    if not ignore_config:
        plot_lens_fit_as_subplot = conf.instance.general.get('output', 'plot_lens_fit_as_subplot', bool)
    else:
        plot_lens_fit_as_subplot = True

    if not plot_lens_fit_as_subplot and ignore_config is False:
        return

    if fit.tracer.total_planes == 1:

        plot_fit_subplot_lens_plane_only(
            fit=fit, should_plot_mask=should_plot_mask, extract_mask_region=extract_mask_region, positions=positions,
            should_plot_image_plane_pix=should_plot_image_plane_pix,
            units=units, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
            output_path=output_path, output_filename=output_filename, output_format=output_format)

    elif fit.tracer.total_planes >= 2:

        plot_fit_subplot_lens_and_source_planes(
            fit=fit, should_plot_mask=should_plot_mask, extract_mask_region=extract_mask_region, positions=positions,
            should_plot_image_plane_pix=should_plot_image_plane_pix,
            units=units, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
            output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_fit_subplot_lens_plane_only(
        fit, should_plot_mask=True, extract_mask_region=False, positions=None, should_plot_image_plane_pix=True,
        units='arcsec', figsize=None, aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10, grid_pointsize=1,
        output_path=None, output_filename='lens_fit', output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=6)

    mask = lens_plotter_util.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

    lens_plotter_util.plot_image(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, positions=positions,
        image_plane_pix_grid=image_plane_pix_grid, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 2)

    lens_plotter_util.plot_noise_map(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, positions=positions, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    lens_plotter_util.plot_signal_to_noise_map(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, positions=positions, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    lens_plotter_util.plot_model_data(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 5)

    lens_plotter_util.plot_residual_map(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 6)

    lens_plotter_util.plot_chi_squared_map(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()

def plot_fit_subplot_lens_and_source_planes(
        fit, should_plot_mask=True, extract_mask_region=False, should_plot_source_grid=False, positions=None,
        should_plot_image_plane_pix=True,
        units='arcsec', figsize=None, aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10, grid_pointsize=1,
        output_path=None, output_filename='lens_fit', output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=9)

    mask = lens_plotter_util.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

    plt.subplot(rows, columns, 1)

    lens_plotter_util.plot_image(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, positions=positions,
        image_plane_pix_grid=image_plane_pix_grid, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 2)

    lens_plotter_util.plot_noise_map(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, positions=None, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    lens_plotter_util.plot_signal_to_noise_map(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, positions=None, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    lens_plotter_util.plot_lens_subtracted_image(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, positions=positions, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title='Fit Lens Subtracted Image', titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        position_pointsize=position_pointsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    if fit.tracer.image_plane.has_light_profile:

        plt.subplot(rows, columns, 5)

        lens_plotter_util.plot_model_image_of_planes(
            fit=fit, plot_foreground=True, mask=mask, extract_mask_region=extract_mask_region, as_subplot=True,
            units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
            title='Fit Image-Plane Model Image', titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 6)

    lens_plotter_util.plot_model_image_of_planes(
        fit=fit, plot_source=True, mask=mask, extract_mask_region=extract_mask_region, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title='Fit Source-Plane Model Image', titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path, output_filename='lens_subtracted_image', output_format=output_format)

    plt.subplot(rows, columns, 7)

    if fit.total_inversions == 0:

        plane_plotters.plot_plane_image(
            plane=fit.tracer.source_plane, positions=None, plot_grid=should_plot_source_grid, as_subplot=True,
            units=units, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
            output_path=output_path, output_filename='', output_format=output_format)

    elif fit.total_inversions == 1:

        inversion_plotters.plot_reconstructed_pixelization(
            inversion=fit.inversion, positions=None, should_plot_grid=False, should_plot_centres=False, as_subplot=True,
            units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(rows, columns, 8)

    lens_plotter_util.plot_residual_map(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 9)

    lens_plotter_util.plot_chi_squared_map(
        fit=fit, mask=mask, extract_mask_region=extract_mask_region, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()

def plot_fit_individuals(fit, should_plot_mask=True, extract_mask_region=False, positions=None,
                         units='kpc',
                         output_path=None, output_format='show', ignore_config=False):

    if fit.tracer.total_planes == 1:

        plot_fit_individuals_lens_plane_only(
            fit=fit, should_plot_mask=should_plot_mask, extract_mask_region=extract_mask_region, positions=positions,
            units=units,
            output_path=output_path, output_format=output_format, ignore_config=ignore_config)

    elif fit.tracer.total_planes == 2:

        plot_fit_individuals_lens_and_source_planes(
            fit=fit, should_plot_mask=should_plot_mask, extract_mask_region=extract_mask_region, positions=positions,
            units=units,
            output_path=output_path, output_format=output_format, ignore_config=ignore_config)

def plot_fit_individuals_lens_plane_only(fit, should_plot_mask=True, extract_mask_region=False, positions=None,
                                         units='kpc',
                                         output_path=None, output_format='show', ignore_config=False):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    if not ignore_config:

        plot_lens_fit_model_image = conf.instance.general.get('output', 'plot_lens_fit_model_image', bool)
        plot_lens_fit_residuals = conf.instance.general.get('output', 'plot_lens_fit_residual_map', bool)
        plot_lens_fit_chi_squareds = conf.instance.general.get('output', 'plot_lens_fit_chi_squared_map', bool)

    else:

        plot_lens_fit_model_image = True
        plot_lens_fit_residuals = True
        plot_lens_fit_chi_squareds = True

    mask = lens_plotter_util.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    if plot_lens_fit_model_image:

        lens_plotter_util.plot_model_data(
            fit=fit, mask=mask, extract_mask_region=extract_mask_region, positions=positions,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if plot_lens_fit_residuals:

        lens_plotter_util.plot_residual_map(
            fit=fit, mask=mask, extract_mask_region=extract_mask_region,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if plot_lens_fit_chi_squareds:

        lens_plotter_util.plot_chi_squared_map(
            fit=fit, mask=mask, extract_mask_region=extract_mask_region,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

def plot_fit_individuals_lens_and_source_planes(
        fit, should_plot_mask=True, extract_mask_region=False, positions=None,
        units='kpc',
        output_path=None, output_format='show', ignore_config=False):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    if not ignore_config:

        plot_lens_fit_lens_subtracted_image = conf.instance.general.get('output', 'plot_lens_fit_lens_subtracted_image',
                                                                        bool)
        plot_lens_fit_model_image = conf.instance.general.get('output', 'plot_lens_fit_model_image', bool)
        plot_lens_fit_lens_model_image = conf.instance.general.get('output', 'plot_lens_fit_lens_model_image', bool)
        plot_lens_fit_source_model_image = conf.instance.general.get('output', 'plot_lens_fit_source_model_image', bool)
        plot_lens_fit_source_plane_image = conf.instance.general.get('output', 'plot_lens_fit_source_plane_image', bool)
        plot_lens_fit_residuals = conf.instance.general.get('output', 'plot_lens_fit_residual_map', bool)
        plot_lens_fit_chi_squareds = conf.instance.general.get('output', 'plot_lens_fit_chi_squared_map', bool)

    else:

        plot_lens_fit_lens_subtracted_image = True
        plot_lens_fit_model_image = True
        plot_lens_fit_lens_model_image = True
        plot_lens_fit_source_model_image = True
        plot_lens_fit_source_plane_image = True
        plot_lens_fit_residuals = True
        plot_lens_fit_chi_squareds = True

    mask = lens_plotter_util.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    if plot_lens_fit_lens_subtracted_image:

        lens_plotter_util.plot_lens_subtracted_image(
            fit=fit, mask=mask, positions=positions, extract_mask_region=extract_mask_region,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if plot_lens_fit_model_image:
        lens_plotter_util.plot_model_data(
            fit=fit, mask=mask, extract_mask_region=extract_mask_region,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if plot_lens_fit_lens_model_image:

        lens_plotter_util.plot_model_image_of_planes(
            fit=fit, mask=mask, extract_mask_region=extract_mask_region, plot_foreground=True,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_filename='fit_lens_plane_model_image', output_format=output_format)

    if plot_lens_fit_source_model_image:

        lens_plotter_util.plot_model_image_of_planes(
            fit=fit, mask=mask, extract_mask_region=extract_mask_region, plot_source=True,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_filename='fit_source_plane_model_image', output_format=output_format)

    if plot_lens_fit_source_plane_image:

        if fit.total_inversions == 0:

           plane_plotters.plot_plane_image(
               plane=fit.tracer.source_plane, plot_grid=True,
               units=units, figsize=(20, 20),
               output_path=output_path, output_filename='fit_source_plane', output_format=output_format)

        elif fit.total_inversions == 1:

            inversion_plotters.plot_reconstructed_pixelization(
                inversion=fit.inversion, should_plot_grid=True,
                units=units, figsize=(20, 20),
                output_path=output_path, output_filename='fit_source_plane', output_format=output_format)

    if plot_lens_fit_residuals:

        lens_plotter_util.plot_residual_map(
            fit=fit, mask=mask, extract_mask_region=extract_mask_region, units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if plot_lens_fit_chi_squareds:

        lens_plotter_util.plot_chi_squared_map(
            fit=fit, mask=mask, extract_mask_region=extract_mask_region, units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)