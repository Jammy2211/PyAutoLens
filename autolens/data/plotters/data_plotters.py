from matplotlib import pyplot as plt

from autolens.data.array.plotters import array_plotters

def plot_image(
        image, plot_origin=True, mask=None, extract_mask_region=False, should_plot_border=False, positions=None,
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, position_pointsize=30, grid_pointsize=1,
        output_path=None, output_format='show', output_filename='image'):
    """Plot the observed image of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

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
        array=image, origin=origin, mask=mask, extract_mask_region=extract_mask_region,
        should_plot_border=should_plot_border, positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_noise_map(
        noise_map, plot_origin=True, mask=None, extract_mask_region=False, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10,
        output_path=None, output_format='show', output_filename='noise_map'):
    """Plot the noise_map-map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    noise_map : ScaledSquarePixelArray
        The noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    origin = get_origin(array=noise_map, plot_origin=plot_origin)

    array_plotters.plot_array(
        array=noise_map, origin=origin, mask=mask, extract_mask_region=extract_mask_region, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_psf(
        psf, plot_origin=True, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='PSF', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='psf'):
    """Plot the PSF of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

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
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_signal_to_noise_map(
        signal_to_noise_map, plot_origin=True, mask=None, extract_mask_region=False, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
        title='Signal-To-Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10,
        output_path=None, output_format='show', output_filename='signal_to_noise_map'):
    """Plot the signal-to-noise_map-map of the ccd data.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    signal_to_noise_map : ScaledSquarePixelArray
        The signal-to-noise map of the data.
    plot_origin : True
        If true, the origin of the data's coordinate system is plotted as a 'x'.
    """
    origin = get_origin(array=signal_to_noise_map, plot_origin=plot_origin)

    array_plotters.plot_array(
        array=signal_to_noise_map, origin=origin, mask=mask, extract_mask_region=extract_mask_region,
        as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
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