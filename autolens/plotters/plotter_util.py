import autofit as af
import matplotlib

backend = af.conf.instance.visualize.get("figures", "backend", str)
matplotlib.use(backend)
import matplotlib.pyplot as plt
import numpy as np

import autoarray as aa
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
        raise exc.PlottingException(
            "Only 1 entry of cb_tick_values or cb_tick_labels was input. You must either supply"
            "both the values and labels, or neither."
        )

    cb.ax.tick_params(labelsize=cb_ticksize)


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

        if output_format is "show":
            plt.show()
        elif output_format is "png":
            plt.savefig(output_path + output_filename + ".png", bbox_inches="tight")
        elif output_format is "fits":
            aa.array_util.numpy_array_2d_to_fits(
                array_2d=array,
                file_path=output_path + output_filename + ".fits",
                overwrite=True,
            )


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
    if output_format is "show":
        plt.show()
    elif output_format is "png":
        plt.savefig(output_path + output_filename + ".png", bbox_inches="tight")
    elif output_format is "fits":
        raise exc.PlottingException("You cannot output a subplots with format .fits")


def get_critical_curve_and_caustic(obj, grid, plot_critical_curve, plot_caustics):

    if plot_critical_curve:
        critical_curves = obj.critical_curves_from_grid(grid=grid)
    else:
        critical_curves = []

    if plot_caustics:
        caustics = obj.caustics_from_grid(grid=grid)
    else:
        caustics = []

    return [critical_curves, caustics]


def plot_lines(line_lists):
    """Plot the liness of the mask or the array on the figure.

    Parameters
    -----------t.
    mask : ndarray of data_type.array.mask.Mask
        The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
    should_plot_lines : bool
        If a mask is supplied, its liness pixels (e.g. the exterior edge) is plotted if this is *True*.
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float or None
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    lines_pointsize : int
        The size of the points plotted to show the liness.
    """
    if line_lists is not None:
        for line_list in line_lists:
            for line in line_list:
                if not line == []:
                    plt.plot(line[:, 1], line[:, 0], c="r", lw=1.5, zorder=200)


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


def radii_bin_size_from_minimum_and_maximum_radii_and_radii_points(
    minimum_radius, maximum_radius, radii_points
):
    return (maximum_radius - minimum_radius) / radii_points


def quantity_radii_from_minimum_and_maximum_radii_and_radii_points(
    minimum_radius, maximum_radius, radii_points
):
    return list(
        np.linspace(start=minimum_radius, stop=maximum_radius, num=radii_points + 1)
    )


def quantity_and_annuli_radii_from_minimum_and_maximum_radii_and_radii_points(
    minimum_radius, maximum_radius, radii_points
):

    radii_bin_size = radii_bin_size_from_minimum_and_maximum_radii_and_radii_points(
        minimum_radius=minimum_radius,
        maximum_radius=maximum_radius,
        radii_points=radii_points,
    )

    quantity_radii = list(
        np.linspace(
            start=minimum_radius + radii_bin_size / 2.0,
            stop=maximum_radius - radii_bin_size / 2.0,
            num=radii_points,
        )
    )
    annuli_radii = list(
        np.linspace(start=minimum_radius, stop=maximum_radius, num=radii_points + 1)
    )

    return quantity_radii, annuli_radii
