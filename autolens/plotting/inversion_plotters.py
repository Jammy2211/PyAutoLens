from autolens import exc
from autolens.inversion import mappers
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import itertools

from astropy.io import fits

def plot_reconstruction(mapper, inversion, as_subplot,
                       units, kpc_per_arcsec,
                       xticks, yticks, xyticksize,
                       norm, norm_min, norm_max, linthresh, linscale,
                       figsize, aspect, cmap, cb_ticksize,
                       title, titlesize, xlabelsize, ylabelsize,
                       output_path, output_filename, output_format):

    pass
    

def plot_rectangular_source(array, points, grid, as_subplot,
               units, kpc_per_arcsec,
               xticks, yticks, xyticksize,
               norm, norm_min, norm_max, linthresh, linscale,
               figsize, aspect, cmap, cb_ticksize,
               title, titlesize, xlabelsize, ylabelsize,
               output_path, output_filename, output_format):

    norm_min, norm_max = get_normalization_min_max(array=array, norm_min=norm_min, norm_max=norm_max)
    norm_scale = get_normalization_scale(norm=norm, norm_min=norm_min, norm_max=norm_max,
                                         linthresh=linthresh, linscale=linscale)

    plot_image(array=array, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm_scale=norm_scale)

    set_title(title=title, titlesize=titlesize)
    set_xy_labels_and_ticks(shape=array.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, xticks=xticks, yticks=yticks,
                            xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    set_colorbar(cb_ticksize=cb_ticksize)
    plot_points(points=points, pointsize=10)
    plot_grid(grid=grid, pointsize=10)

    if not as_subplot:
        output_array(array=array, output_path=output_path, output_filename=output_filename,
                          output_format=output_format)
        plt.close()

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
        raise exc.VisualizeException('The normalization (norm) supplied to the plotter is not a valid string (must be '
                                     'linear | log | symmetric_log')


def plot_image(array, as_subplot, figsize, aspect, cmap, norm_scale):

    if not as_subplot:
        plt.figure(figsize=figsize)

    plt.imshow(np.flipud(array.T), aspect=aspect, cmap=cmap, norm=norm_scale, extent=(0, array.shape[1], 0, array.shape[0]))


def plot_points(points, pointsize):

    if points is not None:
        point_colors = itertools.cycle(["w", "c", "y", "r", "k", "b", "g", "m"])
        for point_set in points:
            plt.scatter(x=point_set[:,0], y=point_set[:,1], color=next(point_colors), s=10.0)


def set_title(title, titlesize):
    plt.title(title, fontsize=titlesize)


def set_xy_labels_and_ticks(shape, units, kpc_per_arcsec, xticks, yticks, xlabelsize, ylabelsize, xyticksize):

    if units is 'pixels':

        plt.xticks(np.round((shape[0] * np.array([0.0, 0.33, 0.66, 0.99]))))
        plt.yticks(np.round((shape[1] * np.array([0.0, 0.33, 0.66, 0.99]))))
        plt.xlabel('x (pixels)', fontsize=xlabelsize)
        plt.ylabel('y (pixels)', fontsize=ylabelsize)

    elif units is 'arcsec' or kpc_per_arcsec is None:

        plt.xticks(shape[0] * np.array([0.0, 0.33, 0.66, 0.99]), np.round(xticks, 1))
        plt.yticks(shape[1] * np.array([0.0, 0.33, 0.66, 0.99]), np.round(yticks, 1))
        plt.xlabel('x (arcsec)', fontsize=xlabelsize)
        plt.ylabel('y (arcsec)', fontsize=ylabelsize)

    elif units is 'kpc':

        plt.xticks(shape[0] * np.array([0.0, 0.33, 0.66, 0.99]), np.round(kpc_per_arcsec * xticks, 1))
        plt.yticks(shape[1] * np.array([0.0, 0.33, 0.66, 0.99]), np.round(kpc_per_arcsec * yticks, 1))
        plt.xlabel('x (kpc)', fontsize=xlabelsize)
        plt.ylabel('y (kpc)', fontsize=ylabelsize)

    else:
        raise exc.VisualizeException('The units supplied to the plotted are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

    plt.tick_params(labelsize=xyticksize)


def set_colorbar(cb_ticksize):
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=cb_ticksize)


def plot_grid(grid, pointsize):

    pass

#   if grid is not None:
#       plt.scatter(x=grid[:, 0], y=grid[:, 1], s=1)


def output_array(array, output_path, output_filename, output_format):
    if output_format is 'show':
        plt.show()
    elif output_format is 'png':
        plt.savefig(output_path + output_filename + '.png', bbox_inches='tight')
    elif output_format is 'fits':
        hdu = fits.PrimaryHDU()
        hdu.data = array
        hdu.writeto(output_path + output_filename + '.fits')


def output_subplot_array(output_path, output_filename, output_format):
    if output_format is 'show':
        plt.show()
    elif output_format is 'png':
        plt.savefig(output_path + output_filename + '.png', bbox_inches='tight')
    elif output_format is 'fits':
        raise exc.VisualizeException('You cannot output a subplots with format .fits')
