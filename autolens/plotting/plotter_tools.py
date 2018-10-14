from autolens import exc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import itertools

from astropy.io import fits

def plot_array(array, as_subplot, figsize, aspect, cmap, norm, norm_max, norm_min, linthresh, linscale):

    if not as_subplot:
        plt.figure(figsize=figsize)

    norm_min, norm_max = get_normalization_min_max(array=array, norm_min=norm_min, norm_max=norm_max)
    norm_scale = get_normalization_scale(norm=norm, norm_min=norm_min, norm_max=norm_max,
                                         linthresh=linthresh, linscale=linscale)

    plt.imshow(array, aspect=aspect, cmap=cmap, norm=norm_scale)

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
        plt.yticks(shape[1] * np.array([0.0, 0.33, 0.66, 0.99]), np.round(-1.0*yticks, 1))
        plt.xlabel('x (arcsec)', fontsize=xlabelsize)
        plt.ylabel('y (arcsec)', fontsize=ylabelsize)

    elif units is 'kpc':

        plt.xticks(shape[0] * np.array([0.0, 0.33, 0.66, 0.99]), np.round(kpc_per_arcsec * xticks, 1))
        plt.yticks(shape[1] * np.array([0.0, 0.33, 0.66, 0.99]), np.round(-1.0*kpc_per_arcsec * yticks, 1))
        plt.xlabel('x (kpc)', fontsize=xlabelsize)
        plt.ylabel('y (kpc)', fontsize=ylabelsize)

    else:
        raise exc.VisualizeException('The units supplied to the plotted are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

    plt.tick_params(labelsize=xyticksize)

def set_colorbar(cb_ticksize):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(cax=cax)
    cb.ax.tick_params(labelsize=cb_ticksize)

def plot_mask(mask):

    if mask is not None:

        plt.gca()
        border_pixels = mask.grid_to_pixel[mask.border_pixels]
        plt.scatter(y=border_pixels[:,0], x=border_pixels[:,1], s=20, c='k')

def plot_points(points):

    if points is not None:
        point_colors = itertools.cycle(["w", "c", "y", "r", "k", "b", "g", "m"])
        for point_set in points:
            plt.scatter(y=point_set[:,0], x=point_set[:,1], color=next(point_colors), s=10.0)

def plot_grid(grid):

    if grid is not None:
        plt.scatter(y=grid[:, 0], x=grid[:, 1], s=1)

def plot_close(as_subplot):

    if not as_subplot:
        plt.close()

def get_subplot_rows_columns_figsize(number_subplots):

    if number_subplots <= 2:
        return 1, 2, (15, 6)
    elif number_subplots <= 4:
        return 2, 2, (13, 10)
    elif number_subplots <= 6:
        return 2,3, (18, 12)
    elif number_subplots <= 9:
        return 3,3, (25, 20)
    elif number_subplots <= 12:
        return 3,4, (25, 20)
    elif number_subplots <= 16:
        return 4,4, (25, 20)
    elif number_subplots <= 20:
        return 4,5, (25, 20)
    else:
        return 6,6, (25, 20)

def output_array(array, as_subplot, output_path, output_filename, output_format):

    if not as_subplot:

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
