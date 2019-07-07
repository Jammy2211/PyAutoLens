import autofit as af
import matplotlib
backend = af.conf.instance.visualize.get('figures', 'backend', str)
matplotlib.use(backend)
from autolens.plotters import array_plotters

def plot_image(
        image, plot_origin=True, grid=None, mask=None, extract_array_from_mask=False, zoom_around_mask=False,
        should_plot_border=False, positions=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, position_pointsize=30, grid_pointsize=1,
        output_path=None, output_format='show', output_filename='image'):
    """Plot the observed image of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or data.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the data, this plots those pixels \
        over the immage.
    """
    origin = get_origin(array=image, plot_origin=plot_origin)

    array_plotters.plot_array(
        array=image, origin=origin, grid=grid, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        should_plot_border=should_plot_border, positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_noise_map(
        noise_map, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10,
        output_path=None, output_format='show', output_filename='noise_map'):
    """Plot the noise_map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    noise_map : ScaledSquarePixelArray
        The noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    origin = get_origin(array=noise_map, plot_origin=plot_origin)

    array_plotters.plot_array(
        array=noise_map, origin=origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_psf(
        psf, plot_origin=True, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='PSF', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='psf'):
    """Plot the PSF of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : ScaledSquarePixelArray
        The psf of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    origin = get_origin(array=psf, plot_origin=plot_origin)

    array_plotters.plot_array(
        array=psf, origin=origin, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_signal_to_noise_map(
        signal_to_noise_map, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False,
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Signal-To-Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10,
        output_path=None, output_format='show', output_filename='signal_to_noise_map'):
    """Plot the signal-to-noise_map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : ScaledSquarePixelArray
        The signal-to-noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    origin = get_origin(array=signal_to_noise_map, plot_origin=plot_origin)

    array_plotters.plot_array(
        array=signal_to_noise_map, origin=origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_absolute_signal_to_noise_map(
        absolute_signal_to_noise_map, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False,
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Absolute Signal-To-Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10,
        output_path=None, output_format='show', output_filename='absolute_signal_to_noise_map'):
    """Plot the absolute signal-to-noise map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    absolute_signal_to_noise_map : ScaledSquarePixelArray
        The absolute signal-to-noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    origin = get_origin(array=absolute_signal_to_noise_map, plot_origin=plot_origin)

    array_plotters.plot_array(
        array=absolute_signal_to_noise_map, origin=origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)
    

def plot_potential_chi_squared_map(
        potential_chi_squared_map, plot_origin=True, mask=None, extract_array_from_mask=False, zoom_around_mask=False,
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Potential Chi-Squared Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10,
        output_path=None, output_format='show', output_filename='potential_chi_squared_map'):
    """Plot the signal-to-noise_map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    potential_chi_squared_map : ScaledSquarePixelArray
        The signal-to-noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    origin = get_origin(array=potential_chi_squared_map, plot_origin=plot_origin)

    array_plotters.plot_array(
        array=potential_chi_squared_map, origin=origin, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        as_subplot=as_subplot,
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