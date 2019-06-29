import autofit as af
import matplotlib
backend = af.conf.instance.visualize.get('figures', 'backend', str)
matplotlib.use(backend)

from matplotlib import pyplot as plt

from autolens.plotters import array_plotters
from autolens.plotters import plotter_util
from autolens.data.plotters import data_plotters


def plot_hyper_galaxy_subplot(
        hyper_galaxy_image, contribution_map, noise_map, hyper_noise_map, chi_squared_map, hyper_chi_squared_map,
        mask=None, extract_array_from_mask=False, zoom_around_mask=False,
        units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10,
        output_path=None, output_format='show', output_filename='hyper_galaxy'):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    
    plt.subplot(rows, columns, 1)

    plot_hyper_galaxy_image(
        hyper_galaxy_image=hyper_galaxy_image, mask=mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 2)

    data_plotters.plot_noise_map(
        noise_map=noise_map, mask=mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 3)

    plot_hyper_noise_map(
        hyper_noise_map=hyper_noise_map, mask=mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)
    
    plt.subplot(rows, columns, 4)

    plot_contribution_map(
        contribution_map=contribution_map, mask=mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 5)

    plot_chi_squared_map(
        chi_squared_map=chi_squared_map, mask=mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 6)

    plot_hyper_chi_squared_map(
        hyper_chi_squared_map=hyper_chi_squared_map, mask=mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


    plotter_util.output_subplot_array(
        output_path=output_path, output_filename=output_filename, output_format=output_format)

    plt.close()

def plot_hyper_galaxy_cluster_images_subplot(
        hyper_galaxy_cluster_image_path_dict, mask,
        should_plot_mask=True, extract_array_from_mask=False, zoom_around_mask=False,
        units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10,
        output_path=None, output_filename='hyper_galaxy_cluster_images', output_format='show'):

        plot_hyper_galaxy_images_subplot(
            hyper_galaxy_image_path_dict=hyper_galaxy_cluster_image_path_dict, mask=mask,
            should_plot_mask=should_plot_mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
            cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
            output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_hyper_galaxy_images_subplot(
        hyper_galaxy_image_path_dict, mask,
        should_plot_mask=True, extract_array_from_mask=False, zoom_around_mask=False,
        units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10,
        output_path=None, output_filename='hyper_galaxy_images', output_format='show'):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=len(hyper_galaxy_image_path_dict))

    if not should_plot_mask:
        mask = False

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    hyper_index = 0

    for path, hyper_galaxy_image in hyper_galaxy_image_path_dict.items():

        hyper_index += 1

        plt.subplot(rows, columns, hyper_index)
        
        plot_hyper_galaxy_image(
            hyper_galaxy_image=hyper_galaxy_image, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, as_subplot=True,
            units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
            cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
            title='Hyper Galaxy = ' + path[0], titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
            output_path=output_path, output_format=output_format, output_filename=output_filename)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()

def plot_hyper_model_image(
        hyper_model_image, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        image_plane_pix_grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Hyper Model Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        grid_pointsize=1, mask_pointsize=10, position_pointsize=10,
        output_path=None, output_format='show', output_filename='hyper_model_image'):
    """Plot the image of a hyper model image.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_model_image : datas.ccd.datas.CCD
        The hyper model image.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=hyper_model_image, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, grid=image_plane_pix_grid,
        positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_hyper_galaxy_image(
        hyper_galaxy_image, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        image_plane_pix_grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Hyper Model Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        grid_pointsize=1, mask_pointsize=10, position_pointsize=10,
        output_path=None, output_format='show', output_filename='hyper_galaxy_image'):
    """Plot the image of a hyper galaxy image.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_galaxy_image : datas.ccd.datas.CCD
        The hyper galaxy image.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=hyper_galaxy_image, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, grid=image_plane_pix_grid,
        positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_contribution_map(
        contribution_map, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        image_plane_pix_grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Contribution Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        grid_pointsize=1, mask_pointsize=10, position_pointsize=10,
        output_path=None, output_format='show', output_filename='contribution_map'):
    """Plot the image of a hyper galaxy image.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    contribution_map : datas.ccd.datas.CCD
        The hyper galaxy image.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=contribution_map, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, grid=image_plane_pix_grid,
        positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_hyper_noise_map(
        hyper_noise_map, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        image_plane_pix_grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Hyper Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        grid_pointsize=1, mask_pointsize=10, position_pointsize=10,
        output_path=None, output_format='show', output_filename='hyper_noise_map'):
    """Plot the image of a hyper galaxy image.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_noise_map : datas.ccd.datas.CCD
        The hyper galaxy image.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=hyper_noise_map, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, grid=image_plane_pix_grid,
        positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_chi_squared_map(
        chi_squared_map, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        image_plane_pix_grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Chi-Squared Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        grid_pointsize=1, mask_pointsize=10, position_pointsize=10,
        output_path=None, output_format='show', output_filename='chi_squared_map'):
    """Plot the image of a hyper galaxy image.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    chi_squared_map : datas.ccd.datas.CCD
        The hyper galaxy image.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=chi_squared_map, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, grid=image_plane_pix_grid,
        positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_hyper_chi_squared_map(
        hyper_chi_squared_map, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        image_plane_pix_grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Hyper Chi-Squared Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        grid_pointsize=1, mask_pointsize=10, position_pointsize=10,
        output_path=None, output_format='show', output_filename='hyper_chi_squared_map'):
    """Plot the image of a hyper galaxy image.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_chi_squared_map : datas.ccd.datas.CCD
        The hyper galaxy image.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotters.plot_array(
        array=hyper_chi_squared_map, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, grid=image_plane_pix_grid,
        positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize, mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)