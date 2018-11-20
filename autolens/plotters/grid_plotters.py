from matplotlib import pyplot as plt
import numpy as np
import itertools

from autolens import exc
from autolens.plotters import plotter_util

def plot_grid(grid, axis_limits=None, points=None, as_subplot=False,
              units='arcsec', kpc_per_arcsec=None,
              figsize=(12, 8), pointsize=5, pointcolor='k', xyticksize=16,
              title='Grid', titlesize=16, xlabelsize=16, ylabelsize=16,
              output_path=None, output_format='show', output_filename='grid'):

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)
    grid = convert_grid_units(grid_arc_seconds=grid, units=units, kpc_per_arcsec=kpc_per_arcsec)
    plt.scatter(y=np.asarray(grid[:, 0]), x=np.asarray(grid[:, 1]), s=pointsize, marker='.')
    plotter_util.set_title(title=title, titlesize=titlesize)
    set_xy_labels_and_ticks_in_arcsec(units, kpc_per_arcsec, grid.xticks, grid.yticks, xlabelsize, ylabelsize,
                                      xyticksize)

    set_axis_limits(axis_limits)
    plot_points(grid, points, pointcolor)

    plt.tick_params(labelsize=xyticksize)
    plotter_util.output_figure(None, as_subplot, output_path, output_filename, output_format)
    plotter_util.close_figure(as_subplot=as_subplot)

def convert_grid_units(grid_arc_seconds, units, kpc_per_arcsec):

    if units is 'arcsec' or kpc_per_arcsec is None:
        return grid_arc_seconds
    elif units is 'kpc':
        return grid_arc_seconds * kpc_per_arcsec

def set_xy_labels_and_ticks_in_arcsec(units, kpc_per_arcsec, xticks, yticks, xlabelsize, ylabelsize, xyticksize):

    if units is 'arcsec' or kpc_per_arcsec is None:

        plt.xlabel('x (arcsec)', fontsize=xlabelsize)
        plt.ylabel('y (arcsec)', fontsize=ylabelsize)

    elif units is 'kpc':

        plt.xlabel('x (kpc)', fontsize=xlabelsize)
        plt.ylabel('y (kpc)', fontsize=ylabelsize)

    else:
        raise exc.PlottingException('The units supplied to the plotted are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

    plt.tick_params(labelsize=xyticksize)

def set_axis_limits(axis_limits):

    if axis_limits is not None:
        plt.axis(axis_limits)

def plot_points(grid, points, pointcolor):

    if points is not None:

        if pointcolor is None:

            point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
            for point_set in points:
                plt.scatter(y=np.asarray(grid[point_set, 0]),
                            x=np.asarray(grid[point_set, 1]), s=8, color=next(point_colors))

        else:

            for point_set in points:
                plt.scatter(y=np.asarray(grid[point_set, 0]),
                            x=np.asarray(grid[point_set, 1]), s=8, color=pointcolor)
