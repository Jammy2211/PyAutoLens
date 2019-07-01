import autofit as af
import matplotlib
backend = af.conf.instance.visualize.get('figures', 'backend', str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autolens.data.plotters import data_plotters
from autolens.plotters import plotter_util

def plot_ccd_subplot(
        ccd_data, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False,
        should_plot_border=False, positions=None,
        units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=30, grid_pointsize=1,
        output_path=None, output_filename='ccd_data', output_format='show'):
    """Plot the ccd data as a sub-plot of all its quantites (e.g. the data, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ccd_data : data.CCDData
        The ccd data, which includes the observed data, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the data, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=6)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_image(
        ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, should_plot_border=should_plot_border, positions=positions, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 2)

    plot_noise_map(
        ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 3)

    plot_psf(
        ccd_data=ccd_data, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 4)

    plot_signal_to_noise_map(
        ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 5)

    plot_absolute_signal_to_noise_map(
        ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 6)

    plot_potential_chi_squared_map(
        ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()


def plot_ccd_individual(
        ccd_data, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        should_plot_image=False,
        should_plot_noise_map=False,
        should_plot_psf=False,
        should_plot_signal_to_noise_map=False,
        should_plot_absolute_signal_to_noise_map=False,
        should_plot_potential_chi_squared_map=False,
        units='arcsec',
        output_path=None, output_format='png'):
    """Plot each attribute of the ccd data as individual figures one by one (e.g. the data, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    ccd_data : data.CCDData
        The ccd data, which includes the observed data, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    if should_plot_image:

        plot_image(
            ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, positions=positions,
            units=units,
            output_path=output_path, output_format=output_format)

    if should_plot_noise_map:

        plot_noise_map(
            ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format=output_format)

    if should_plot_psf:

        plot_psf(
            ccd_data=ccd_data, plot_origin=plot_origin,
            output_path=output_path, output_format=output_format)

    if should_plot_signal_to_noise_map:

        plot_signal_to_noise_map(
            ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format=output_format)

    if should_plot_absolute_signal_to_noise_map:

        plot_absolute_signal_to_noise_map(
            ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format=output_format)
    
    if should_plot_potential_chi_squared_map:

        plot_potential_chi_squared_map(
            ccd_data=ccd_data, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format=output_format)

def plot_image(
        ccd_data, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False,
        should_plot_border=False, positions=None,
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='CCD Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, position_pointsize=30, grid_pointsize=1,
        output_path=None, output_format='show', output_filename='ccd_image'):
    """Plot the observed data of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data.CCDData
        The ccd data, which includes the observed data, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the data, this plots those pixels \
        over the immage.
    """
    data_plotters.plot_image(
        image=ccd_data.image, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, should_plot_border=should_plot_border, positions=positions,
        as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_noise_map(
        ccd_data, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='CCD Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16, mask_pointsize=10,
        output_path=None, output_format='show', output_filename='ccd_noise_map'):
    """Plot the noise_map-map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data.CCDData
        The ccd data, which includes the observed data, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    data_plotters.plot_noise_map(
        noise_map=ccd_data.noise_map, plot_origin=plot_origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_psf(
        ccd_data, plot_origin=True, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='CCD PSF', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='ccd_psf'):
    """Plot the PSF of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data.CCDData
        The ccd data, which includes the observed data, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    data_plotters.plot_psf(
        psf=ccd_data.psf, plot_origin=plot_origin, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_signal_to_noise_map(
        ccd_data, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='CCD Signal-To-Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10,
        output_path=None, output_format='show', output_filename='ccd_signal_to_noise_map'):
    """Plot the signal-to-noise_map-map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data.CCDData
        The ccd data, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    data_plotters.plot_signal_to_noise_map(
        signal_to_noise_map=ccd_data.signal_to_noise_map, plot_origin=plot_origin, mask=mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_absolute_signal_to_noise_map(
        ccd_data, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='CCD Absolute Signal-To-Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10,
        output_path=None, output_format='show', output_filename='ccd_absolute_signal_to_noise_map'):
    """Plot the signal-to-noise_map-map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data.CCDData
        The ccd data, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    data_plotters.plot_absolute_signal_to_noise_map(
        absolute_signal_to_noise_map=ccd_data.absolute_signal_to_noise_map, plot_origin=plot_origin, mask=mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_potential_chi_squared_map(
        ccd_data, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='CCD Potential Chi-Squared Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10,
        output_path=None, output_format='show', output_filename='ccd_potential_chi_squared_map'):
    """Plot the signal-to-noise_map-map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : data.CCDData
        The ccd data, which includes the observed image, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """

    data_plotters.plot_potential_chi_squared_map(
        potential_chi_squared_map=ccd_data.potential_chi_squared_map, plot_origin=plot_origin, mask=mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def get_origin(array, plot_origin):
    """Get the (y,x) origin of the ccd data if it going to be plotted.

    Parameters
    -----------
    array : data.array.scaled_array.ScaledArray
        The array from which the origin is extracted.
    plot_origin : True
        If true, the origin of the data's coordinate system is returned.
    """
    if plot_origin:
        return array.origin
    else:
        return None