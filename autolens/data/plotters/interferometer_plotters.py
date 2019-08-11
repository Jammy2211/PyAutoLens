import autofit as af
import matplotlib

backend = af.conf.instance.visualize.get("figures", "backend", str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autolens.data.plotters import data_plotters
from autolens.plotters import plotter_util

import numpy as np

def plot_interferometer_subplot(
    interferometer_data,
    plot_origin=True,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    plot_axis_type="linear",
    legend_fontsize=12,
    output_path=None,
    output_filename="interferometer_data",
    output_format="show",
):
    """Plot the interferometer instrument as a sub-plot of all its quantites (e.g. the instrument, noise_map-map, PSF, Signal-to_noise-map, \
     etc).

    Set *autolens.instrument.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer_data : instrument.InterferometerData
        The interferometer instrument, which includes the observed instrument, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the instrument's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or instrument.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the instrument, this plots those pixels \
        over the immage.
    ignore_config : bool
        If *False*, the config file general.ini is used to determine whether the subpot is plotted. If *True*, the \
        config file is ignored.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=3
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_visibilities(
        interferometer_data=interferometer_data,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

    plt.subplot(rows, columns, 2)

    plot_u_wavelengths(
        interferometer_data=interferometer_data,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        plot_axis_type=plot_axis_type,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        legend_fontsize=legend_fontsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename
    )

    plt.subplot(rows, columns, 4)

    plot_v_wavelengths(
        interferometer_data=interferometer_data,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        plot_axis_type=plot_axis_type,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        legend_fontsize=legend_fontsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename
    )

    plt.subplot(rows, columns, 3)

    plot_primary_beam(
        interferometer_data=interferometer_data,
        plot_origin=plot_origin,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
    )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def plot_interferometer_individual(
    interferometer_data,
    should_plot_visibilities=False,
    should_plot_u_wavelengths=False,
    should_plot_v_wavelengths=False,
    should_plot_primary_beam=False,
    units="arcsec",
    output_path=None,
    output_format="png",
):
    """Plot each attribute of the interferometer instrument as individual figures one by one (e.g. the instrument, noise_map-map, PSF, \
     Signal-to_noise-map, etc).

    Set *autolens.instrument.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    interferometer_data : instrument.InterferometerData
        The interferometer instrument, which includes the observed instrument, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the instrument's coordinate system is plotted as a 'x'.
    """

    if should_plot_visibilities:

        plot_visibilities(
            interferometer_data=interferometer_data,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_u_wavelengths:

        plot_u_wavelengths(
            interferometer_data=interferometer_data,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_v_wavelengths:

        plot_v_wavelengths(
            interferometer_data=interferometer_data,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_primary_beam:

        plot_primary_beam(
            interferometer_data=interferometer_data,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )


def plot_visibilities(
    interferometer_data,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=(7, 7),
    cmap="jet",
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Visibilities",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="interferometer_visibilities",
):
    """Plot the observed image of the ccd instrument.

    Set *autolens.instrument.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the instrument.
    plot_origin : True
        If true, the origin of the instrument's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or instrument.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the instrument, this plots those pixels \
        over the immage.
    """

    data_plotters.plot_visibilities(
        real_visibilities=interferometer_data.real_visibilities,
        imaginary_visibilities=interferometer_data.imaginary_visibilities,
        visibilities_noise_map=interferometer_data.visibilities_noise_map,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )

def plot_u_wavelengths(
    interferometer_data,
    as_subplot=False,
    label='Wavelengths',
    units="",
    kpc_per_arcsec=None,
    figsize=(14, 7),
    plot_axis_type="linear",
    ylabel="U-Wavelength",
    title="U-Wavelengths",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    legend_fontsize=12,
    output_path=None,
    output_format="show",
    output_filename="interferometer_u_wavelengths",
):
    """Plot the observed image of the ccd instrument.

    Set *autolens.instrument.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the instrument.
    plot_origin : True
        If true, the origin of the instrument's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or instrument.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the instrument, this plots those pixels \
        over the immage.
    """

    data_plotters.plot_u_wavelengths(
        u_wavelengths=interferometer_data.u_wavelengths,
        as_subplot=as_subplot,
        label=label,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        plot_axis_type=plot_axis_type,
        ylabel=ylabel,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        legend_fontsize=legend_fontsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename
    )

def plot_v_wavelengths(
    interferometer_data,
    as_subplot=False,
    label='Wavelengths',
    units="",
    kpc_per_arcsec=None,
    figsize=(14, 7),
    plot_axis_type="linear",
    ylabel="V-Wavelength",
    title="V-Wavelengths",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    legend_fontsize=12,
    output_path=None,
    output_format="show",
    output_filename="interferometer_v_wavelengths",
):
    """Plot the observed image of the ccd instrument.

    Set *autolens.instrument.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : ScaledSquarePixelArray
        The image of the instrument.
    plot_origin : True
        If true, the origin of the instrument's coordinate system is plotted as a 'x'.
    image_plane_pix_grid : ndarray or instrument.array.grid_stacks.PixGrid
        If an adaptive pixelization whose pixels are formed by tracing pixels from the instrument, this plots those pixels \
        over the immage.
    """

    data_plotters.plot_v_wavelengths(
        v_wavelengths=interferometer_data.v_wavelengths,
        as_subplot=as_subplot,
        label=label,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        plot_axis_type=plot_axis_type,
        ylabel=ylabel,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        legend_fontsize=legend_fontsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename
    )

def plot_primary_beam(
    interferometer_data,
    plot_origin=True,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="CCD PSF",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="interferometer_primary_beam",
):
    """Plot the PSF of the interferometer instrument.

    Set *autolens.instrument.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    image : instrument.CCDData
        The interferometer instrument, which includes the observed instrument, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the instrument's coordinate system is plotted as a 'x'.
    """

    data_plotters.plot_primary_beam(
        primary_beam=interferometer_data.primary_beam,
        plot_origin=plot_origin,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )