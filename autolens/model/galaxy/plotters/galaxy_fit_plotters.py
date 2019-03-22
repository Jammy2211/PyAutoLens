from matplotlib import pyplot as plt

from autolens import exc
from autofit import conf
from autolens.model.galaxy import galaxy_data as gd
from autolens.data.array.plotters import plotter_util, array_plotters
from autolens.lens.plotters import lens_plotter_util


def plot_fit_subplot(
        fit, should_plot_mask=True, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
        output_path=None, output_filename='galaxy_fit', output_format='show'):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=4)

    mask = lens_plotter_util.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_galaxy_data_array(
        galaxy_data=fit.galaxy_data, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename=output_filename, output_format=output_format)

    plt.subplot(rows, columns, 2)

    lens_plotter_util.plot_model_data(
        fit=fit, mask=mask, extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
        positions=positions, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title='Model Galaxy', titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    lens_plotter_util.plot_residual_map(
        fit=fit, mask=mask, extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
        as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    lens_plotter_util.plot_chi_squared_map(
        fit=fit, mask=mask, extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
        as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                               output_format=output_format)

    plt.close()

def plot_fit_individuals(
        fit, should_plot_mask=True, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        should_plot_image=False,
        should_plot_noise_map=False,
        should_plot_model_image=False,
        should_plot_residual_map=False,
        should_plot_chi_squared_map=False,
        units='kpc',
        output_path=None, output_format='show'):

    mask = lens_plotter_util.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    kpc_per_arcsec = None

    if should_plot_image:

        plot_galaxy_data_array(
            galaxy_data=fit.galaxy_data, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, positions=positions,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if should_plot_noise_map:

        lens_plotter_util.plot_noise_map(
            fit=fit, mask=mask, extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            positions=positions,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if should_plot_model_image:

        lens_plotter_util.plot_model_data(
            fit=fit, mask=mask, extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            positions=positions,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if should_plot_residual_map:

        lens_plotter_util.plot_residual_map(
            fit=fit, mask=mask, extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

    if should_plot_chi_squared_map:

        lens_plotter_util.plot_chi_squared_map(
            fit=fit, mask=mask, extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path, output_format=output_format)

def plot_galaxy_data_array(
        galaxy_data, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
        output_path=None, output_filename='galaxy_data', output_format='show'):

    if galaxy_data.use_intensities:
        title='Galaxy Data Intensities'
    elif galaxy_data.use_convergence:
        title='Galaxy Data Convergence'
    elif galaxy_data.use_potential:
        title='Galaxy Data Potential'
    elif galaxy_data.use_deflections_y:
        title='Galaxy Data Deflections (y)'
    elif galaxy_data.use_deflections_x:
        title='Galaxy Data Deflections (x)'
    else:
        raise exc.PlottingException('The galaxy data array does not have a True use_profile_type')

    array_plotters.plot_array(
        array=galaxy_data.image, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions,
        as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)