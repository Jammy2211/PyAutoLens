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

def get_normalization(norm, norm_min, norm_max, linthresh, linscale):

    if norm is 'linear':
        return colors.Normalize(vmin=norm_min, vmax=norm_max)
    elif norm is 'log':
        return colors.LogNorm(vmin=norm_min, vmax=norm_max)
    elif norm is 'symmetric_log':
        return colors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=norm_min, vmax=norm_max)
    else:
        raise exc.VisualizeException('The normalization (norm) supplied to the plotter is not a valid string (must be '
                                     'linear | log | symmetric_log')

def set_title_and_labels(title, xlabel, ylabel, title_size=56, xlabel_size=56, ylabel_size=56):

    plt.title(title, fontsize=title_size)
    plt.xlabel(xlabel, fontsize=xlabel_size)
    plt.ylabel(ylabel, fontsize=ylabel_size)

def set_ticks(array, xticks, yticks):

    plt.xticks(array.shape[0] * np.array([0.0, 0.33, 0.66, 0.99]), xticks)
    plt.yticks(array.shape[1] * np.array([0.0, 0.33, 0.66, 0.99]), yticks)
    plt.tick_params(labelsize=50)

def set_colorbar(norm_min, norm_max):

    cb = plt.colorbar(ticks=[norm_min, 0.0, norm_max], format=LogFormatter(labelOnlyBase=False))
    cb.ax.set_yticklabels([np.round(norm_min, 2), 0.0, np.round(norm_max, 2)])
    cb.ax.tick_params(labelsize=32)

def output_array(array, output_path, output_filename, output_type):

    if output_type is 'show':
        plt.show()
    elif output_type is 'png':
        plt.savefig(output_path + output_filename, bbox_inches='tight')
    elif output_type is 'fits':
        hdu = fits.PrimaryHDU()
        hdu.data = array
        hdu.writeto(output_path + output_filename)