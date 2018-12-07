import matplotlib.pyplot as plt
from astropy.io import fits

from autolens import exc


def get_subplot_rows_columns_figsize(number_subplots):
    """Get the size of a sub plot in (rows, columns), based on the number of subplots that are going to be plotted.

    Parameters
    -----------
    number_subplots : int
        The number of subplots that are to be plotted in the figure.
    """
    if number_subplots <= 2:
        return 1, 2, (18, 8)
    elif number_subplots <= 4:
        return 2, 2, (13, 10)
    elif number_subplots <= 6:
        return 2, 3, (18, 12)
    elif number_subplots <= 9:
        return 3, 3, (25, 20)
    elif number_subplots <= 12:
        return 3, 4, (25, 20)
    elif number_subplots <= 16:
        return 4, 4, (25, 20)
    elif number_subplots <= 20:
        return 4, 5, (25, 20)
    else:
        return 6, 6, (25, 20)


def setup_figure(figsize, as_subplot):
    """Setup a figure for plotting an image.

    Parameters
    -----------
    figsize : (int, int)
        The size of the figure in (rows, columns).
    as_subplot : bool
        If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
        new figure.
    """
    if not as_subplot:
        plt.figure(figsize=figsize)


def set_title(title, titlesize):
    plt.title(title, fontsize=titlesize)


def output_figure(array, as_subplot, output_path, output_filename, output_format):
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
        raise exc.PlottingException('You cannot output a subplots with format .fits')


def close_figure(as_subplot):
    if not as_subplot:
        plt.close()
