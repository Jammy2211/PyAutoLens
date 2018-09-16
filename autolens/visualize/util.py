from autolens import exc
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter
import numpy as np

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

def get_xylabels(units):

    if units is 'arcsec':
        xlabel = 'x (arcsec)'
        ylabel = 'y (arcsec)'
    elif units is 'kpc':
        xlabel = 'x (kpc)'
        ylabel = 'y (kpc)'
    else:
        raise exc.VisualizeException('The units supplied to the plotted are not a valid string (must be arcsec | kpc)')

    return xlabel, ylabel

def plot_image(array, figsize, aspect, cmap, norm_scale):
    plt.figure(figsize=figsize)
    plt.imshow(array, aspect=aspect, cmap=cmap, norm=norm_scale)

def set_title_and_labels(title, xlabel, ylabel, titlesize, xlabelsize, ylabelsize):

    plt.title(title, fontsize=titlesize)
    plt.xlabel(xlabel, fontsize=xlabelsize)
    plt.ylabel(ylabel, fontsize=ylabelsize)

def set_ticks(array, xticks, yticks, xyticksize):

    plt.xticks(array.shape[0] * np.array([0.0, 0.33, 0.66, 0.99]), xticks)
    plt.yticks(array.shape[1] * np.array([0.0, 0.33, 0.66, 0.99]), yticks)
    plt.tick_params(labelsize=xyticksize)

def set_colorbar(cb_ticksize):

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=cb_ticksize)

def output_array(array, output_path, output_filename, output_format):

    if output_format is 'show':
        plt.show()
    elif output_format is 'png':
        plt.savefig(output_path + output_filename + '.png', bbox_inches='tight')
    elif output_format is 'fits':
        hdu = fits.PrimaryHDU()
        hdu.data = array
        hdu.writeto(output_path + output_filename + '.fits')