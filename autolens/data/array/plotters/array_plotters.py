from autolens import exc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import itertools

from autolens.data.array.plotters import plotter_util


def plot_array(array, origin=None, mask=None, extract_array_from_mask=False, zoom_around_mask=False,
               should_plot_border=False, positions=None, grid=None, as_subplot=False,
               units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
               title='Array', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
               mask_pointsize=10, border_pointsize=2, position_pointsize=30, grid_pointsize=1,
               xticks_manual=None, yticks_manual=None,
               output_path=None, output_format='show', output_filename='array'):
    """Plot an array of data as a figure.

    Parameters
    -----------
    array : data.array.scaled_array.ScaledArray
        The 2D array of data which is plotted.
    origin : (float, float).
        The origin of the coordinate system of the array, which is plotted as an 'x' on the image if input.
    mask : data.array.mask.Mask
        The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
    extract_array_from_mask : bool
        The plotter array is extracted using the mask, such that masked values are plotted as zeros. This ensures \
        bright features outside the mask do not impact the color map of the plot.
    zoom_around_mask : bool
        If True, the 2D region of the array corresponding to the rectangle encompassing all unmasked values is \
        plotted, thereby zooming into the region of interest.
    should_plot_border : bool
        If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is *True*.
    positions : [[]]
        Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
    grid : data.array.grids.RegularGrid
        A grid of (y,x) coordinates which may be plotted over the plotted array.
    as_subplot : bool
        Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    figsize : (int, int)
        The size of the figure in (rows, columns).
    aspect : str
        The aspect ratio of the array, specifically whether it is forced to be square ('equal') or adapts its size to \
        the figure size ('auto').
    cmap : str
        The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
    norm : str
        The normalization of the colormap used to plot the image, specifically whether it is linear ('linear'), log \
        ('log') or a symmetric log normalization ('symmetric_log').
    norm_min : float or None
        The minimum array value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum array value the colormap map spans (all values above this value are plotted the same color).
    linthresh : float
        For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
        is linear.
    linscale : float
        For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
        relative to the logarithmic range.
    cb_ticksize : int
        The size of the tick labels on the colorbar.
    cb_fraction : float
        The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
    cb_pad : float
        Pads the color bar in the figure, which resizes the colorbar relative to the figure.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    mask_pointsize : int
        The size of the points plotted to show the mask.
    border_pointsize : int
        The size of the points plotted to show the borders.
    positions_pointsize : int
        The size of the points plotted to show the input positions.
    grid_pointsize : int
        The size of the points plotted to show the grid.
    xticks_manual :  [] or None
        If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
    yticks_manual :  [] or None
        If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
    output_path : str
        The path on the hard-disk where the figure is output.
    output_filename : str
        The filename of the figure that is output.
    output_format : str
        The format the figue is output:
        'show' - display on computer screen.
        'png' - output to hard-disk as a png.
        'fits' - output to hard-disk as a fits file.'

    Returns
    --------
    None

    Examples
    --------
        array_plotters.plot_array(
        array=image, origin=(0.0, 0.0), mask=circular_mask, extract_array_from_mask=True, zoom_around_mask=True,
        should_plot_border=False, positions=[[1.0, 1.0], [2.0, 2.0]], grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7,7), aspect='auto',
        cmap='jet', norm='linear, norm_min=None, norm_max=None, linthresh=None, linscale=None,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, border_pointsize=2, position_pointsize=10, grid_pointsize=10,
        xticks_manual=None, yticks_manual=None,
        output_path='/path/to/output', output_format='png', output_filename='image')
    """

    if array is None:
        return

    if extract_array_from_mask and mask is not None:
        array = np.add(array, 0.0, out=np.zeros_like(array), where=np.asarray(mask) == 0)

    if zoom_around_mask and mask is not None:
        array = array.extract_scaled_array_around_mask(mask=mask, buffer=2)

    plot_figure(array=array, as_subplot=as_subplot, units=units, kpc_per_arcsec=kpc_per_arcsec,
                figsize=figsize, aspect=aspect, cmap=cmap, norm=norm,
                norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
                xticks_manual=xticks_manual, yticks_manual=yticks_manual)

    plotter_util.set_title(title=title, titlesize=titlesize)
    set_xy_labels_and_ticksize(units=units, kpc_per_arcsec=kpc_per_arcsec, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                               xyticksize=xyticksize)

    set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                 cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels)
    plot_origin(array=array, origin=origin, units=units, kpc_per_arcsec=kpc_per_arcsec)
    plot_mask(mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec, pointsize=mask_pointsize)
    plot_border(mask=mask, should_plot_border=should_plot_border, units=units, kpc_per_arcsec=kpc_per_arcsec,
                pointsize=border_pointsize)
    plot_points(points_arcsec=positions, array=array, units=units, kpc_per_arcsec=kpc_per_arcsec,
                pointsize=position_pointsize)
    plot_grid(grid_arcsec=grid, array=array, units=units, kpc_per_arcsec=kpc_per_arcsec, pointsize=grid_pointsize)
    plotter_util.output_figure(array, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                               output_format=output_format)
    plotter_util.close_figure(as_subplot=as_subplot)

def plot_figure(array, as_subplot, units, kpc_per_arcsec, figsize, aspect, cmap, norm, norm_min, norm_max,
                linthresh, linscale, xticks_manual, yticks_manual):
    """Open a matplotlib figure and plot the array of data on it.

    Parameters
    -----------
    array : data.array.scaled_array.ScaledArray
        The 2D array of data which is plotted.
    as_subplot : bool
        Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    figsize : (int, int)
        The size of the figure in (rows, columns).
    aspect : str
        The aspect ratio of the array, specifically whether it is forced to be square ('equal') or adapts its size to \
        the figure size ('auto').
    cmap : str
        The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
    norm : str
        The normalization of the colormap used to plot the image, specifically whether it is linear ('linear'), log \
        ('log') or a symmetric log normalization ('symmetric_log').
    norm_min : float or None
        The minimum array value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum array value the colormap map spans (all values above this value are plotted the same color).
    linthresh : float
        For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
        is linear.
    linscale : float
        For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
        relative to the logarithmic range.
    xticks_manual :  [] or None
        If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
    yticks_manual :  [] or None
        If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
    """

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    norm_min, norm_max = get_normalization_min_max(array=array, norm_min=norm_min, norm_max=norm_max)
    norm_scale = get_normalization_scale(norm=norm, norm_min=norm_min, norm_max=norm_max,
                                         linthresh=linthresh, linscale=linscale)

    extent = get_extent(array=array, units=units, kpc_per_arcsec=kpc_per_arcsec,
                        xticks_manual=xticks_manual, yticks_manual=yticks_manual)

    plt.imshow(array, aspect=aspect, cmap=cmap, norm=norm_scale, extent=extent)

def get_extent(array, units, kpc_per_arcsec, xticks_manual, yticks_manual):
    """Get the extent of the dimensions of the array in the units of the figure (e.g. arc-seconds or kpc).

    This is used to set the extent of the array and thus the y / x axis limits.

    Parameters
    -----------
    array : data.array.scaled_array.ScaledArray
        The 2D array of data which is plotted.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    xticks_manual :  [] or None
        If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
    yticks_manual :  [] or None
        If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
    """
    if xticks_manual is not None and yticks_manual is not None:
        return np.asarray([xticks_manual[0], xticks_manual[3], yticks_manual[0], yticks_manual[3]])

    if units in 'pixels':
        return np.asarray([0, array.shape[1], 0, array.shape[0]])
    elif units in 'arcsec' or kpc_per_arcsec is None:
        return np.asarray([array.arc_second_minima[1], array.arc_second_maxima[1],
                           array.arc_second_minima[0], array.arc_second_maxima[0]])
    elif units in 'kpc':
        return list(map(lambda tick : tick*kpc_per_arcsec,
                        np.asarray([array.arc_second_minima[1], array.arc_second_maxima[1],
                                    array.arc_second_minima[0], array.arc_second_maxima[0]])))
    else:
        raise exc.PlottingException('The units supplied to the plotted are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

def get_normalization_min_max(array, norm_min, norm_max):
    """Get the minimum and maximum of the normalization of the array, which sets the lower and upper limits of the \
    colormap.

    If norm_min / norm_max are not supplied, the minimum / maximum values of the array of data are used.

    Parameters
    -----------
    array : data.array.scaled_array.ScaledArray
        The 2D array of data which is plotted.
    norm_min : float or None
        The minimum array value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum array value the colormap map spans (all values above this value are plotted the same color).
    """
    if norm_min is None:
        norm_min = array.min()
    if norm_max is None:
        norm_max = array.max()

    return norm_min, norm_max

def get_normalization_scale(norm, norm_min, norm_max, linthresh, linscale):
    """Get the normalization scale of the colormap. This will be scaled based on the input min / max normalization \
    values.

    For a 'symmetric_log' colormap, linthesh and linscale also change the colormap.

    If norm_min / norm_max are not supplied, the minimum / maximum values of the array of data are used.

    Parameters
    -----------
    array : data.array.scaled_array.ScaledArray
        The 2D array of data which is plotted.
    norm_min : float or None
        The minimum array value the colormap map spans (all values below this value are plotted the same color).
    norm_max : float or None
        The maximum array value the colormap map spans (all values above this value are plotted the same color).
    linthresh : float
        For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
        is linear.
    linscale : float
        For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
        relative to the logarithmic range.
    """
    if norm is 'linear':
        return colors.Normalize(vmin=norm_min, vmax=norm_max)
    elif norm is 'log':
        if norm_min == 0.0:
            norm_min = 1.e-4
        return colors.LogNorm(vmin=norm_min, vmax=norm_max)
    elif norm is 'symmetric_log':
        return colors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=norm_min, vmax=norm_max)
    else:
        raise exc.PlottingException('The normalization (norm) supplied to the plotter is not a valid string (must be '
                                     'linear | log | symmetric_log')

def set_xy_labels_and_ticksize(units, kpc_per_arcsec, xlabelsize, ylabelsize, xyticksize):
    """Set the x and y labels of the figure, and set the fontsize of those labels.

    The x and y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depend on the \
    units the figure is plotted in.

    Parameters
    -----------
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    """

    if units in 'pixels':

        plt.xlabel('x (pixels)', fontsize=xlabelsize)
        plt.ylabel('y (pixels)', fontsize=ylabelsize)

    elif units in 'arcsec' or kpc_per_arcsec is None:

        plt.xlabel('x (arcsec)', fontsize=xlabelsize)
        plt.ylabel('y (arcsec)', fontsize=ylabelsize)

    elif units in 'kpc':

        plt.xlabel('x (kpc)', fontsize=xlabelsize)
        plt.ylabel('y (kpc)', fontsize=ylabelsize)

    else:
        raise exc.PlottingException('The units supplied to the plotter are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

    plt.tick_params(labelsize=xyticksize)

def set_colorbar(cb_ticksize, cb_fraction, cb_pad, cb_tick_values, cb_tick_labels):
    """Setup the colorbar of the figure, specifically its ticksize and the size is appears relative to the figure.

    Parameters
    -----------
    cb_ticksize : int
        The size of the tick labels on the colorbar.
    cb_fraction : float
        The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
    cb_pad : float
        Pads the color bar in the figure, which resizes the colorbar relative to the figure.
    cb_tick_values : [float]
        Manually specified values of where the colorbar tick labels appear on the colorbar.
    cb_tick_labels : [float]
        Manually specified labels of the color bar tick labels, which appear where specified by cb_tick_values.
    """

    if cb_tick_values is None and cb_tick_labels is None:
        cb = plt.colorbar(fraction=cb_fraction, pad=cb_pad)
    elif cb_tick_values is not None and cb_tick_labels is not None:
        cb = plt.colorbar(fraction=cb_fraction, pad=cb_pad, ticks=cb_tick_values)
        cb.ax.set_yticklabels(cb_tick_labels)
    else:
        raise exc.PlottingException('Only 1 entry of cb_tick_values or cb_tick_labels was input. You must either supply'
                                    'both the values and labels, or neither.')

    cb.ax.tick_params(labelsize=cb_ticksize)

def convert_grid_units(array, grid_arcsec, units, kpc_per_arcsec):
    """Convert the grid from its input units (arc-seconds) to the input unit (e.g. retain arc-seconds) or convert to \
    another set of units (pixels or kilo parsecs).

    Parameters
    -----------
    array : data.array.scaled_array.ScaledArray
        The 2D array of data which is plotted, the shape of which is used for converting the grid to units of pixels.
    grid_arcsec : ndarray or data.array.grids.RegularGrid
        The (y,x) coordinates of the grid in arc-seconds, in an array of shape (total_coordinates, 2).
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    """
    if units in 'pixels':
        return array.grid_arcsec_to_grid_pixels(grid_arcsec=grid_arcsec)
    elif units in 'arcsec' or kpc_per_arcsec is None:
        return grid_arcsec
    elif units in 'kpc':
        return grid_arcsec * kpc_per_arcsec
    else:
        raise exc.PlottingException('The units supplied to the plotter are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

def plot_origin(array, origin, units, kpc_per_arcsec):
    """Plot the (y,x) origin ofo the array's coordinates as a 'x'.
    
    Parameters
    -----------
    array : data.array.scaled_array.ScaledArray
        The 2D array of data which is plotted.
    origin : (float, float).
        The origin of the coordinate system of the array, which is plotted as an 'x' on the image if input.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    """
    if origin is not None:

        origin_grid = np.asarray(origin)
        origin_units = convert_grid_units(array=array, grid_arcsec=origin_grid, units=units,
                                          kpc_per_arcsec=kpc_per_arcsec)
        plt.scatter(y=origin_units[0], x=origin_units[1], s=80, c='k', marker='x')

def plot_mask(mask, units, kpc_per_arcsec, pointsize):
    """Plot the mask of the array on the figure.

    Parameters
    -----------
    mask : ndarray of data.array.mask.Mask
        The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    pointsize : int
        The size of the points plotted to show the mask.
    """

    if mask is not None:

        plt.gca()
        edge_pixels = mask.masked_grid_index_to_pixel[mask.edge_pixels] + 0.5
        edge_pixels[:,0] -= mask.extraction_offset[0] + 1
        edge_pixels[:,1] -= mask.extraction_offset[1] + 1
        edge_arcsec = mask.grid_pixels_to_grid_arcsec(grid_pixels=edge_pixels)
        edge_units = convert_grid_units(array=mask, grid_arcsec=edge_arcsec, units=units,
                                          kpc_per_arcsec=kpc_per_arcsec)

        plt.scatter(y=edge_units[:,0], x=edge_units[:,1], s=pointsize, c='k')

def plot_border(mask, should_plot_border, units, kpc_per_arcsec, pointsize):
    """Plot the borders of the mask or the array on the figure.

    Parameters
    -----------t.
    mask : ndarray of data.array.mask.Mask
        The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
    should_plot_border : bool
        If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is *True*.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    border_pointsize : int
        The size of the points plotted to show the borders.
    """
    if should_plot_border and mask is not None:

        plt.gca()
        border_pixels = mask.masked_grid_index_to_pixel[mask.border_pixels]
        border_arcsec = mask.grid_pixels_to_grid_arcsec(grid_pixels=border_pixels)
        border_units = convert_grid_units(array=mask, grid_arcsec=border_arcsec, units=units,
                                          kpc_per_arcsec=kpc_per_arcsec)

        plt.scatter(y=border_units[:,0], x=border_units[:,1], s=pointsize, c='y')

def plot_points(points_arcsec, array, units, kpc_per_arcsec, pointsize):
    """Plot a set of points over the array of data on the figure.

    Parameters
    -----------
    positions : [[]]
        Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
    array : data.array.scaled_array.ScaledArray
        The 2D array of data which is plotted.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    pointsize : int
        The size of the points plotted to show the input positions.
    """
    if points_arcsec is not None:
        points_arcsec = list(map(lambda position_set: np.asarray(position_set), points_arcsec))
        point_colors = itertools.cycle(["m", "y", "r", "w", "c", "b", "g", "k"])
        for point_set_arcsec in points_arcsec:
            point_set_units = convert_grid_units(array=array, grid_arcsec=point_set_arcsec, units=units,
                                                 kpc_per_arcsec=kpc_per_arcsec)
            plt.scatter(y=point_set_units[:,0], x=point_set_units[:,1], color=next(point_colors), s=pointsize)

def plot_grid(grid_arcsec, array, units, kpc_per_arcsec, pointsize):
    """Plot a grid of points over the array of data on the figure.

     Parameters
     -----------.
     grid_arcsec : ndarray or data.array.grids.RegularGrid
         A grid of (y,x) coordinates in arc-seconds which may be plotted over the array.
     array : data.array.scaled_array.ScaledArray
        The 2D array of data which is plotted.
     units : str
         The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
     kpc_per_arcsec : float or None
         The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
     grid_pointsize : int
         The size of the points plotted to show the grid.
     """
    if grid_arcsec is not None:
        grid_units = convert_grid_units(grid_arcsec=grid_arcsec, array=array, units=units,
                                        kpc_per_arcsec=kpc_per_arcsec)

        plt.scatter(y=np.asarray(grid_units[:, 0]), x=np.asarray(grid_units[:, 1]), s=pointsize, c='k')