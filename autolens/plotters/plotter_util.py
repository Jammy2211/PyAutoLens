import matplotlib.pyplot as plt
from astropy.io import fits

from autolens import exc
from autolens.data.array.util import array_util

import numpy as np

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
        new figure and so that it can be output using the *output_subplot_array* function.
    """
    if not as_subplot:
        fig = plt.figure(figsize=figsize)
        return fig


def set_title(title, titlesize):
    """Set the title and title size of the figure.

    Parameters
    -----------
    title : str
        The text of the title.
    titlesize : int
        The size of of the title of the figure.
    """
    plt.title(title, fontsize=titlesize)


def output_figure(array, as_subplot, output_path, output_filename, output_format):
    """Output the figure, either as an image on the screen or to the hard-disk as a .png or .fits file.

    Parameters
    -----------
    array : ndarray
        The 2D array of image to be output, required for outputting the image as a fits file.
    as_subplot : bool
        Whether the figure is part of subplot, in which case the figure is not output so that the entire subplot can \
        be output instead using the *output_subplot_array* function.
    output_path : str
        The path on the hard-disk where the figure is output.
    output_filename : str
        The filename of the figure that is output.
    output_format : str
        The format the figue is output:
        'show' - display on computer screen.
        'png' - output to hard-disk as a png.
        'fits' - output to hard-disk as a fits file.'
    """
    if not as_subplot:

        if output_format is 'show':
            plt.show()
        elif output_format is 'png':
            plt.savefig(output_path + output_filename + '.png', bbox_inches='tight')
        elif output_format is 'fits':
            array_util.numpy_array_2d_to_fits(array_2d=array, file_path=output_path + output_filename + '.fits',
                                              overwrite=True)


def output_subplot_array(output_path, output_filename, output_format):
    """Output a figure which consists of a set of subplot,, either as an image on the screen or to the hard-disk as a \
    .png file.

    Parameters
    -----------
    output_path : str
        The path on the hard-disk where the figure is output.
    output_filename : str
        The filename of the figure that is output.
    output_format : str
        The format the figue is output:
        'show' - display on computer screen.
        'png' - output to hard-disk as a png.
    """
    if output_format is 'show':
        plt.show()
    elif output_format is 'png':
        plt.savefig(output_path + output_filename + '.png', bbox_inches='tight')
    elif output_format is 'fits':
        raise exc.PlottingException('You cannot output a subplots with format .fits')


def close_figure(as_subplot):
    """After plotting and outputting a figure, close the matplotlib figure instance (omit if a subplot).

    Parameters
    -----------
    as_subplot : bool
        Whether the figure is part of subplot, in which case the figure is not closed so that the entire figure can \
        be closed later after output.
    """
    if not as_subplot:
        plt.close()


# def check_units_distance_can_be_plotted(units_distance, kpc_per_arcsec):


def radii_bin_size_from_minimum_and_maximum_radii_and_radii_points(minimum_radius, maximum_radius, radii_points):
    return (maximum_radius - minimum_radius) / radii_points


def quantity_radii_from_minimum_and_maximum_radii_and_radii_points(minimum_radius, maximum_radius, radii_points):
    return list(np.linspace(start=minimum_radius, stop=maximum_radius, num=radii_points + 1))


def quantity_and_annuli_radii_from_minimum_and_maximum_radii_and_radii_points(minimum_radius, maximum_radius, radii_points):

    radii_bin_size = radii_bin_size_from_minimum_and_maximum_radii_and_radii_points(
        minimum_radius=minimum_radius, maximum_radius=maximum_radius, radii_points=radii_points)

    quantity_radii = list(np.linspace(start=minimum_radius + radii_bin_size/2.0, stop=maximum_radius - radii_bin_size/2.0,
                                      num=radii_points))
    annuli_radii = list(np.linspace(start=minimum_radius, stop=maximum_radius, num=radii_points + 1))

    return quantity_radii, annuli_radii