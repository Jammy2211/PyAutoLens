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
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
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
    mask_pointsize=10,
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
        mask=mask,
        include_critical_curves=include_critical_curves,
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
        mask_pointsize=mask_pointsize,
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
            mask=mask,
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
            mask=mask,
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

    if tracer.has_mass_profile:

        caustics = plotter_util.get_critical_curve_and_caustic(
            obj=tracer,
            grid=grid,
            include_critical_curves=False,
            include_caustics=include_caustics,
        )

    else:

        caustics = None

    plane_plotters.plane_image(
        plane=tracer.source_plane,
        grid=source_plane_grid,
        lines=caustics,
        as_subplot=True,
        positions=None,
        include_grid=False,
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
            mask=mask,
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
            mask=mask,
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
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    positions=None,
    plot_profile_image=False,
    plot_source_plane=False,
    plot_convergence=False,
    plot_potential=False,
    plot_deflections=False,
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

    if plot_profile_image:

        profile_image(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include_critical_curves=include_critical_curves,
            positions=positions,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_convergence:

        convergence(
            tracer=tracer,
            grid=grid,
            mask=mask,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_potential:

        potential(
            tracer=tracer,
            grid=grid,
            mask=mask,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_source_plane:

        source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[-1]

        if tracer.has_mass_profile:

            caustics = plotter_util.get_critical_curve_and_caustic(
                obj=tracer,
                grid=grid,
                include_critical_curves=False,
                include_caustics=include_caustics,
            )

        else:

            caustics = None

        plane_plotters.plane_image(
            plane=tracer.source_plane,
            grid=source_plane_grid,
            lines=caustics,
            positions=None,
            include_grid=False,
            units=units,
            output_path=output_path,
            output_filename="tracer_source_plane",
            output_format=output_format,
        )

    if plot_deflections:

        deflections_y(
            tracer=tracer,
            grid=grid,
            mask=mask,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_deflections:

        deflections_x(
            tracer=tracer,
            grid=grid,
            mask=mask,
            units=units,
            output_path=output_path,
            output_format=output_format,
        )


def profile_image(
    tracer,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
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
    title="Tracer Profile Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10.0,
    output_path=None,
    output_format="show",
    output_filename="tracer_profile_image",
):

    profile_image = tracer.profile_image_from_grid(grid=grid)

    if tracer.has_mass_profile:

        lines = plotter_util.get_critical_curve_and_caustic(
            obj=tracer,
            grid=grid,
            include_critical_curves=include_critical_curves,
            include_caustics=include_caustics,
        )

    else:

        lines = None

    aa.plot.array(
        array=profile_image,
        mask=mask,
        lines=lines,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def convergence(
    tracer,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
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

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=tracer,
        grid=grid,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    aa.plot.array(
        array=convergence,
        mask=mask,
        lines=lines,
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
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
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

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=tracer,
        grid=grid,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    aa.plot.array(
        array=potential,
        mask=mask,
        lines=lines,
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
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
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

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=tracer,
        grid=grid,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    aa.plot.array(
        array=deflections_y,
        mask=mask,
        lines=lines,
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
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
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

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=tracer,
        grid=grid,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    aa.plot.array(
        array=deflections_x,
        lines=lines,
        mask=mask,
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
