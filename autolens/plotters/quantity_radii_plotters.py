from autolens import exc
from autolens.plotters import plotter_util

import matplotlib.pyplot as plt

def plot_quantity_as_function_of_radius(
        quantity, radii, as_subplot=False, label=None, plot_axis_type='semilogy',
        effective_radius_line=None, einstein_radius_line=None,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), plot_legend=True,
        title='Quantity vs Radius', ylabel='Quantity', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        legend_fontsize=12,
        output_path=None, output_format='show', output_filename='quantity_vs_radius'):

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)
    plotter_util.set_title(title=title, titlesize=titlesize)

    set_xy_labels_and_ticksize(units=units, kpc_per_arcsec=kpc_per_arcsec, ylabel=ylabel, xlabelsize=xlabelsize,
                               ylabelsize=ylabelsize, xyticksize=xyticksize)

    plot_quantity_vs_radius_data(quantity=quantity, radii=radii, plot_axis_type=plot_axis_type, label=label)
    plot_vertical_line(units=units, kpc_per_arcsec=kpc_per_arcsec, vertical_line_x_value=effective_radius_line,
                        label='Effective Radius')
    plot_vertical_line(units=units, kpc_per_arcsec=kpc_per_arcsec, vertical_line_x_value=einstein_radius_line,
                        label='Einstein Radius')

    set_legend(plot_legend=plot_legend, legend_fontsize=legend_fontsize)
    plotter_util.output_figure(array=None, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                               output_format=output_format)
    plotter_util.close_figure(as_subplot=as_subplot)

def plot_quantity_vs_radius_data(quantity, radii, plot_axis_type, label):

    if plot_axis_type is 'linear':
        plt.plot(radii, quantity, label=label)
    elif plot_axis_type is 'semilogy':
        plt.semilogy(radii, quantity, label=label)
    elif plot_axis_type is 'loglog':
        plt.loglog(radii, quantity, label=label)
    else:
        raise exc.PlottingException('The plot_axis_type supplied to the plotter is not a valid string (must be linear '
                                    '| semilogy | loglog)')

def plot_vertical_line(units, kpc_per_arcsec, vertical_line_x_value, label):

    if vertical_line_x_value is None:
        return

    if units in 'arcsec' or kpc_per_arcsec is None:
        x_value_plot = vertical_line_x_value
    elif units in 'kpc':
        x_value_plot = vertical_line_x_value * kpc_per_arcsec
    else:
        raise exc.PlottingException('The units supplied to the plotter are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

    plt.axvline(x=x_value_plot, label=label, linestyle='--')

def set_xy_labels_and_ticksize(units, kpc_per_arcsec, ylabel, xlabelsize, ylabelsize, xyticksize):
    """Set the x and y labels of the figure, and set the fontsize of those labels.

    The x label is always the distance scale / radius, thus the x-label is either arc-seconds or kpc and depending \
    on the units the figure is plotted in.

    The ylabel is the physical quantity being plotted and is passed as an input parameter.

    Parameters
    -----------
    units : str
        The units of the y / x axis of the plots, in arc-seconds ('arcsec') or kiloparsecs ('kpc').
    kpc_per_arcsec : float
        The conversion factor between arc-seconds and kiloparsecs, required to plot the units in kpc.
    ylabel : str
        The y-label of the figure, which is the physical quantitiy being plotted.
    xlabelsize : int
        The fontsize of the x axes label.
    ylabelsize : int
        The fontsize of the y axes label.
    xyticksize : int
        The font size of the x and y ticks on the figure axes.
    """

    plt.ylabel(ylabel=ylabel, fontsize=ylabelsize)

    if units in 'arcsec' or kpc_per_arcsec is None:

        plt.xlabel('x (arcsec)', fontsize=xlabelsize)

    elif units in 'kpc':

        plt.xlabel('x (kpc)', fontsize=xlabelsize)

    else:
        raise exc.PlottingException('The units supplied to the plotter are not a valid string (must be pixels | '
                                     'arcsec | kpc)')

    plt.tick_params(labelsize=xyticksize)

def set_legend(plot_legend, legend_fontsize):
    if plot_legend:
        plt.legend(fontsize=legend_fontsize)