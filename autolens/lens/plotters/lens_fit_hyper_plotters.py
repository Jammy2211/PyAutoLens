from matplotlib import pyplot as plt

from autofit import conf
from autolens.data.array.plotters import plotter_util
from autolens.lens.plotters import lens_plotter_util

def plot_fit_subplot(
        fit_hyper, fit, should_plot_mask=True, zoom_around_mask=False, positions=None, 
        should_plot_image_plane_pix=False,
        units='arcsec', figsize=None, aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10, grid_pointsize=1,
        output_path=None, output_filename='lens_fit', output_format='show'):

    if fit.tracer.total_planes == 1:

        plot_fit_subplot_lens_plane_only(
            fit_hyper=fit_hyper, fit=fit, should_plot_mask=should_plot_mask, zoom_around_mask=zoom_around_mask, 
            positions=positions, should_plot_image_plane_pix=should_plot_image_plane_pix,
            units=units, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
            output_path=output_path, output_filename=output_filename, output_format=output_format)


def plot_fit_subplot_lens_plane_only(
        fit_hyper, fit, should_plot_mask=True, zoom_around_mask=False, positions=None,
        should_plot_image_plane_pix=False,
        units='arcsec', figsize=None, aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10, grid_pointsize=1,
        output_path=None, output_filename='lens_hyper_fit', output_format='show'):
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

    mask = lens_plotter_util.get_mask(fit=fit_hyper, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid(
        should_plot_image_plane_pix=should_plot_image_plane_pix, fit=fit)

    lens_plotter_util.plot_image(
        fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask, positions=positions,
        image_plane_pix_grid=image_plane_pix_grid, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 2)

    lens_plotter_util.plot_model_data(
        fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
         units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
         cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
         cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
         titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
         output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    lens_plotter_util.plot_residual_map(
        fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 5)

    lens_plotter_util.plot_chi_squared_map(
        fit=fit, mask=mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.subplot(rows, columns, 4)

    lens_plotter_util.plot_contribution_maps(
        fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 6)

    lens_plotter_util.plot_chi_squared_map(
        fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 8)

    lens_plotter_util.plot_noise_map(
        fit=fit, mask=mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 9)

    lens_plotter_util.plot_noise_map(
        fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()


def plot_fit_individuals(fit_hyper, fit, should_plot_mask=True, zoom_around_mask=False, positions=None,
                         should_plot_image_plane_pix=False,
                         should_plot_image=False, should_plot_noise_map=False,
                         should_plot_lens_subtracted_image=False,
                         should_plot_model_image=False,
                         should_plot_lens_model_image=False,
                         should_plot_source_model_image=False,
                         should_plot_source_plane_image=False,
                         should_plot_contribution_map=False,
                         should_plot_residual_map=False,
                         should_plot_chi_squared_map=False,
                         units='kpc',
                         output_path=None, output_format='show'):

    if fit.tracer.total_planes == 1:

        plot_fit_individuals_lens_plane_only(
            fit_hyper=fit_hyper, fit=fit, should_plot_mask=should_plot_mask, zoom_around_mask=zoom_around_mask,
            positions=positions, should_plot_image_plane_pix=should_plot_image_plane_pix,
            should_plot_image=should_plot_image,
            should_plot_noise_map=should_plot_noise_map,
            should_plot_model_image=should_plot_model_image,
            should_plot_contribution_map=should_plot_contribution_map,
            should_plot_residual_map=should_plot_residual_map,
            should_plot_chi_squared_map=should_plot_chi_squared_map,
            units=units,
            output_path=output_path, output_format=output_format)

def plot_fit_individuals_lens_plane_only(
        fit_hyper, fit, should_plot_mask=True, zoom_around_mask=False, positions=None,
        should_plot_image_plane_pix=False,
        should_plot_image=False,
        should_plot_noise_map=False,
        should_plot_model_image=False,
        should_plot_contribution_map=False,
        should_plot_residual_map=False,
        should_plot_chi_squared_map=False,
        units='kpc',
        output_path=None, output_format='show'):
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

    mask = lens_plotter_util.get_mask(fit=fit_hyper, should_plot_mask=should_plot_mask)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    if should_plot_image:

        image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

        lens_plotter_util.plot_image(
            fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask,
            image_plane_pix_grid=image_plane_pix_grid,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_filename='fit_hyper_image', output_format=output_format)

    if should_plot_noise_map:

        lens_plotter_util.plot_noise_map(
            fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_filename='fit_hyper_noise_map', output_format=output_format)

    if should_plot_model_image:

        lens_plotter_util.plot_model_data(
            fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask, positions=positions,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if should_plot_residual_map:

        lens_plotter_util.plot_residual_map(
            fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if should_plot_chi_squared_map:

        lens_plotter_util.plot_chi_squared_map(
            fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if should_plot_contribution_map:

        lens_plotter_util.plot_contribution_maps(
            fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if should_plot_chi_squared_map:

        lens_plotter_util.plot_chi_squared_map(
            fit=fit_hyper, mask=mask, zoom_around_mask=zoom_around_mask,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_filename='fit_hyper_chi_squared_map', output_format=output_format)