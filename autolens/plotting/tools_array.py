from autolens import exc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import itertools

from autolens.plotting import tools

def plot_array(array, mask=None, positions=None, grid=None, as_subplot=False,
               units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
               title='Array', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
               mask_pointsize=10, position_pointsize=30, grid_pointsize=1,
               xticks_manual=None, yticks_manual=None,
               output_path=None, output_format='show', output_filename='array'):

    plot_figure(array=array, as_subplot=as_subplot, units=units, kpc_per_arcsec=kpc_per_arcsec,
                figsize=figsize, aspect=aspect, cmap=cmap, norm=norm,
                norm_max=norm_max, norm_min=norm_min, linthresh=linthresh, linscale=linscale,
                xticks_manual=xticks_manual, yticks_manual=yticks_manual)
    tools.set_title(title=title, titlesize=titlesize)
    set_xy_labels_and_ticksize(units=units, kpc_per_arcsec=kpc_per_arcsec, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                               xyticksize=xyticksize)

    set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    plot_mask(mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec, pointsize=mask_pointsize)
    plot_points(points_arc_seconds=positions, array=array, units=units, kpc_per_arcsec=kpc_per_arcsec,
                pointsize=position_pointsize)
    plot_grid(grid_arc_seconds=grid, array=array, units=units, kpc_per_arcsec=kpc_per_arcsec, pointsize=grid_pointsize)
    tools.output_figure(array, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_figure(array, as_subplot, units, kpc_per_arcsec, figsize, aspect, cmap, norm, norm_max, norm_min,
                linthresh, linscale, xticks_manual, yticks_manual):

    tools.setup_figure(figsize=figsize, as_subplot=as_subplot)

    norm_min, norm_max = get_normalization_min_max(array=array, norm_min=norm_min, norm_max=norm_max)
    norm_scale = get_normalization_scale(norm=norm, norm_min=norm_min, norm_max=norm_max,
                                         linthresh=linthresh, linscale=linscale)

    extent = get_extent(array=array, units=units, kpc_per_arcsec=kpc_per_arcsec,
                        xticks_manual=xticks_manual, yticks_manual=yticks_manual)

    plt.imshow(array, aspect=aspect, cmap=cmap, norm=norm_scale, extent=extent)

def get_extent(array, units, kpc_per_arcsec, xticks_manual, yticks_manual):

    if xticks_manual is not None and yticks_manual is not None:
        return [xticks_manual[0], xticks_manual[3], yticks_manual[0], yticks_manual[3]]

    if units is 'pixels':
        return [0, array.shape[1], 0, array.shape[0]]
    elif units is 'arcsec' or kpc_per_arcsec is None:
        return [array.xticks[0], array.xticks[3], array.yticks[0], array.yticks[3]]
    elif units is 'kpc':
        return list(map(lambda tick : tick*kpc_per_arcsec,
                        [array.xticks[0], array.xticks[3], array.yticks[0], array.yticks[3]]))

def get_normalization_min_max(array, norm_min, norm_max):

    if norm_min is None:
        norm_min = array.min()
    if norm_max is None:
        norm_max = array.max()

    return norm_min, norm_max

def get_normalization_scale(norm, norm_min, norm_max, linthresh, linscale):
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

    if units is 'pixels':

        plt.xlabel('x (pixels)', fontsize=xlabelsize)
        plt.ylabel('y (pixels)', fontsize=ylabelsize)

    elif units is 'arcsec' or kpc_per_arcsec is None:

        plt.xlabel('x (arcsec)', fontsize=xlabelsize)
        plt.ylabel('y (arcsec)', fontsize=ylabelsize)

    elif units is 'kpc':

        plt.xlabel('x (kpc)', fontsize=xlabelsize)
        plt.ylabel('y (kpc)', fontsize=ylabelsize)

    else:
        raise exc.PlottingException('The units supplied to the plotted are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

    plt.tick_params(labelsize=xyticksize)

def set_colorbar(cb_ticksize, cb_fraction, cb_pad):

    cb = plt.colorbar(fraction=cb_fraction, pad=cb_pad)
    cb.ax.tick_params(labelsize=cb_ticksize)

def convert_grid_units(array, grid_arc_seconds, units, kpc_per_arcsec):

    if units is 'pixels':
        return array.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=grid_arc_seconds)
    elif units is 'arcsec' or kpc_per_arcsec is None:
        return grid_arc_seconds
    elif units is 'kpc':
        return grid_arc_seconds * kpc_per_arcsec

def plot_mask(mask, units, kpc_per_arcsec, pointsize):

    if mask is not None:

        plt.gca()
        border_pixels = mask.grid_to_pixel[mask.border_pixels]
        border_arc_seconds = mask.grid_pixels_to_grid_arc_seconds(grid_pixels=border_pixels)
        border_units = convert_grid_units(array=mask, grid_arc_seconds=border_arc_seconds, units=units,
                                          kpc_per_arcsec=kpc_per_arcsec)

        plt.scatter(y=border_units[:,0], x=border_units[:,1], s=pointsize, c='k')

def plot_points(points_arc_seconds, array, units, kpc_per_arcsec, pointsize):

    if points_arc_seconds is not None:
        points_arc_seconds = list(map(lambda position_set: np.asarray(position_set), points_arc_seconds))
        point_colors = itertools.cycle(["m", "y", "r", "w", "c", "b", "g", "k"])
        for point_set_arc_seconds in points_arc_seconds:
            point_set_units = convert_grid_units(array=array, grid_arc_seconds=point_set_arc_seconds, units=units,
                                                 kpc_per_arcsec=kpc_per_arcsec)
            plt.scatter(y=point_set_units[:,0], x=point_set_units[:,1], color=next(point_colors), s=pointsize)

def plot_grid(grid_arc_seconds, array, units, kpc_per_arcsec, pointsize):

    if grid_arc_seconds is not None:
        grid_units = convert_grid_units(grid_arc_seconds=grid_arc_seconds, array=array, units=units,
                                        kpc_per_arcsec=kpc_per_arcsec)

        plt.scatter(y=np.asarray(grid_units[:, 0]), x=np.asarray(grid_units[:, 1]), s=pointsize, c='k')