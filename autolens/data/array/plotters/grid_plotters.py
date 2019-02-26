from matplotlib import pyplot as plt
import numpy as np
import itertools

from autolens import exc
from autolens.data.array.plotters import plotter_util


def plot_grid(grid, axis_limits=None, points=None, as_subplot=False,
              units='arcsec', kpc_per_arcsec=None,
              figsize=(12, 8), pointsize=5, pointcolor='k', xyticksize=16,
              title='Grid', titlesize=16, xlabelsize=16, ylabelsize=16,
              output_path=None, output_format='show', output_filename='grid'):
    """Plot a grid of (y,x) Cartesian coordinates as a scatter plot of points.

    Parameters
    -----------
    grid : data.array.grids.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
    axis_limits : []
        The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
    points : []
        A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
        different planes).
    as_subplot : bool
        Whether the grid is plotted as part of a subplot, in which case the grid figure is not opened / closed.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    figsize : (int, int)
        The size of the figure in (rows, columns).
    pointsize : int
        The size of the points plotted on the grid.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    title : str
        The text of the title.
    titlesize : int
        The size of of the title of the figure.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    output_path : str
        The path on the hard-disk where the figure is output.
    output_filename : str
        The filename of the figure that is output.
    output_format : str
        The format the figue is output:
        'show' - display on computer screen.
        'png' - output to hard-disk as a png.
    """

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)
    grid = convert_grid_units(grid_arcsec=grid, units=units, kpc_per_arcsec=kpc_per_arcsec)
    plt.scatter(y=np.asarray(grid[:, 0]), x=np.asarray(grid[:, 1]), s=pointsize, marker='.')
    plotter_util.set_title(title=title, titlesize=titlesize)
    set_xy_labels(units, kpc_per_arcsec, xlabelsize, ylabelsize, xyticksize)

    set_axis_limits(axis_limits)
    plot_points(grid, points, pointcolor)

    plt.tick_params(labelsize=xyticksize)
    plotter_util.output_figure(None, as_subplot, output_path, output_filename, output_format)
    plotter_util.close_figure(as_subplot=as_subplot)

def convert_grid_units(grid_arcsec, units, kpc_per_arcsec):
    """Convert the grid from its input units (arc-seconds) to the input unit (e.g. retain arc-seconds) or convert to \
    another set of units (kiloparsecs).

    Parameters
    -----------
    grid_arcsec : ndarray or data.array.grids.RegularGrid
        The (y,x) coordinates of the grid in arc-seconds, in an array of shape (total_coordinates, 2).
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    """

    if units in 'arcsec' or kpc_per_arcsec is None:
        return grid_arcsec
    elif units in 'kpc':
        return grid_arcsec * kpc_per_arcsec

def set_xy_labels(units, kpc_per_arcsec, xlabelsize, ylabelsize, xyticksize):
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
    if units in 'arcsec' or kpc_per_arcsec is None:

        plt.xlabel('x (arcsec)', fontsize=xlabelsize)
        plt.ylabel('y (arcsec)', fontsize=ylabelsize)

    elif units in 'kpc':

        plt.xlabel('x (kpc)', fontsize=xlabelsize)
        plt.ylabel('y (kpc)', fontsize=ylabelsize)

    else:
        raise exc.PlottingException('The units supplied to the plotted are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

    plt.tick_params(labelsize=xyticksize)

def set_axis_limits(axis_limits):
    """Set the axis limits of the figure the grid is plotted on.

    Parameters
    -----------
    axis_limits : []
        The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
    """
    if axis_limits is not None:
        plt.axis(axis_limits)

def plot_points(grid, points, pointcolor):
    """Plot a subset of points in a different color, to highlight a specifc region of the grid (e.g. how certain \
    pixels map between different planes).

    Parameters
    -----------
    grid : ndarray or data.array.grids.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
    points : []
        A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
        different planes).
    pointcolor : str or None
        The color the points should be plotted. If None, the points are iterated through a cycle of colors.
    """
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
