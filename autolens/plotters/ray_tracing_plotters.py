import autofit as af
import matplotlib

backend = af.conf.instance.visualize.get("figures", "backend", str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

import autoarray as aa
from autoarray.plotters import plotter_util
from autolens.plotters import plane_plotters


def subplot(
    tracer,
    grid,
    mask_overlay=None,
    positions=None,
    units="arcsec",
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
    mask_overlay_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1.0,
    output_path=None,
    output_filename="tracer",
    output_format="show",
):
    """Plot the observed _tracer of an analysis, using the *Imaging* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    tracer : autolens.imaging.tracer.Imaging
        Class containing the _tracer,  noise_map-mappers and PSF that are to be plotted.
        The font size of the figure ylabel.
    output_path : str
        The path where the _tracer is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _tracer is output. File formats (e.g. png, fits) output the _tracer to harddisk. 'show' displays the _tracer \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    profile_image(
        tracer=tracer,
        grid=grid,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=True,
        units=units,
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
        mask_overlay_pointsize=mask_overlay_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    if tracer.has_mass_profile:

        plt.subplot(rows, columns, 2)

        convergence(
            tracer=tracer,
            grid=grid,
            mask_overlay=mask_overlay,
            as_subplot=True,
            units=units,
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
            output_filename="",
            output_format=output_format,
        )

        plt.subplot(rows, columns, 3)

        potential(
            tracer=tracer,
            grid=grid,
            mask_overlay=mask_overlay,
            as_subplot=True,
            units=units,
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
            output_filename="",
            output_format=output_format,
        )

    plt.subplot(rows, columns, 4)

    source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[-1]

    plane_plotters.plane_image(
        plane=tracer.source_plane,
        grid=source_plane_grid,
        as_subplot=True,
        positions=None,
        plot_grid=False,
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
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    if tracer.has_mass_profile:

        plt.subplot(rows, columns, 5)

        deflections_y(
            tracer=tracer,
            grid=grid,
            mask_overlay=mask_overlay,
            as_subplot=True,
            units=units,
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
            output_filename="",
            output_format=output_format,
        )

        plt.subplot(rows, columns, 6)

        deflections_x(
            tracer=tracer,
            grid=grid,
            mask_overlay=mask_overlay,
            as_subplot=True,
            units=units,
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
            output_filename="",
            output_format=output_format,
        )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def individual(
    tracer,
    grid,
    mask_overlay=None,
    positions=None,
    should_plot_profile_image=False,
    should_plot_source_plane=False,
    should_plot_convergence=False,
    should_plot_potential=False,
    should_plot_deflections=False,
    units="arcsec",
    output_path=None,
    output_format="show",
):
    """Plot the observed _tracer of an analysis, using the *Imaging* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    tracer : autolens.imaging.tracer.Imaging
        Class containing the _tracer, noise_map-mappers and PSF that are to be plotted.
        The font size of the figure ylabel.
    output_path : str
        The path where the _tracer is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _tracer is output. File formats (e.g. png, fits) output the _tracer to harddisk. 'show' displays the _tracer \
        in the python interpreter window.
    """

    if should_plot_profile_image:

        profile_image(
            tracer=tracer,
            grid=grid,
            mask_overlay=mask_overlay,
            positions=positions,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_convergence:

        convergence(
            tracer=tracer,
            grid=grid,
            mask_overlay=mask_overlay,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_potential:

        potential(
            tracer=tracer,
            grid=grid,
            mask_overlay=mask_overlay,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_source_plane:

        source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[-1]

        plane_plotters.plane_image(
            plane=tracer.source_plane,
            grid=source_plane_grid,
            positions=None,
            plot_grid=False,
            units=units,
            output_path=output_path,
            output_filename="tracer_source_plane",
            output_format=output_format,
        )

    if should_plot_deflections:

        deflections_y(
            tracer=tracer,
            grid=grid,
            mask_overlay=mask_overlay,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_deflections:

        deflections_x(
            tracer=tracer,
            grid=grid,
            mask_overlay=mask_overlay,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )


def profile_image(
    tracer,
    grid,
    mask_overlay=None,
    positions=None,
    as_subplot=False,
    units="arcsec",
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
    title="Tracer Imaging-Plane Imaging",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_overlay_pointsize=10,
    position_pointsize=10.0,
    output_path=None,
    output_format="show",
    output_filename="tracer_profile_image",
):

    profile_image = tracer.profile_image_from_grid(grid=grid)

    aa.plot.array(
        array=profile_image,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec,
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
        mask_overlay_pointsize=mask_overlay_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def convergence(
    tracer,
    grid,
    mask_overlay=None,
    as_subplot=False,
    units="arcsec",
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
    title="Tracer Convergence",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="tracer_convergence",
):

    convergence = tracer.convergence_from_grid(grid=grid)

    aa.plot.array(
        array=convergence,
        mask_overlay=mask_overlay,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec,
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


def potential(
    tracer,
    grid,
    mask_overlay=None,
    as_subplot=False,
    units="arcsec",
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
    title="Tracer Potential",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="tracer_potential",
):

    potential = tracer.potential_from_grid(grid=grid)

    aa.plot.array(
        array=potential,
        mask_overlay=mask_overlay,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec,
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


def deflections_y(
    tracer,
    grid,
    mask_overlay=None,
    as_subplot=False,
    units="arcsec",
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
    title="Tracer Deflections (y)",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="tracer_deflections_y",
):

    deflections = tracer.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_from_sub_array_1d(sub_array_1d=deflections[:, 0])

    aa.plot.array(
        array=deflections_y,
        mask_overlay=mask_overlay,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec,
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


def deflections_x(
    tracer,
    grid,
    mask_overlay=None,
    as_subplot=False,
    units="arcsec",
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
    title="Tracer Deflections (x)",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="tracer_deflections_x",
):

    deflections = tracer.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_from_sub_array_1d(sub_array_1d=deflections[:, 1])

    aa.plot.array(
        array=deflections_x,
        mask_overlay=mask_overlay,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec,
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
