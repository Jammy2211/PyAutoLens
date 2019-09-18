import autofit as af
import matplotlib

backend = af.conf.instance.visualize.get("figures", "backend", str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autolens.plotters import plotter_util, grid_plotters, array_plotters


def plot_profile_image(
    plane,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
    plot_grid=False,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Plane Profile Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="plane_profile_image",
):

    profile_image = plane.profile_image_from_grid(
        grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    )

    if plane.has_mass_profile:
        lines = plotter_util.get_critical_curve_and_caustic(
            obj=plane,
            grid=grid,
            plot_critical_curve=plot_critical_curves,
            plot_caustics=plot_caustics,
        )
    else:
        lines = None

    if not plot_grid:
        grid = None

    array_plotters.plot_array(
        array=profile_image,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        grid=grid,
        lines=lines,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=plane.kpc_per_arcsec,
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
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_plane_image(
    plane,
    grid,
    plot_origin=True,
    positions=None,
    plot_grid=True,
    lines=None,
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
    title="Plane Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="plane_plane_image",
):

    plane_image = plane.plane_image_from_grid(grid=grid)

    if not plot_grid:
        grid = None

    if plot_origin:
        origin = plane_image.origin
    else:
        origin = None

    array_plotters.plot_array(
        array=plane_image,
        origin=origin,
        positions=positions,
        grid=grid,
        lines=lines,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=plane.kpc_per_arcsec,
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
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_convergence(
    plane,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Plane Convergence",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="plane_convergence",
):

    convergence = plane.convergence_from_grid(
        grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    )

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=plane,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=convergence,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        lines=lines,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=plane.kpc_per_arcsec,
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


def plot_potential(
    plane,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Plane Potential",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="plane_potential",
):

    potential = plane.potential_from_grid(
        grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    )

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=plane,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=potential,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        lines=lines,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=plane.kpc_per_arcsec,
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


def plot_deflections_y(
    plane,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Plane Deflections (y)",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="plane_deflections_y",
):

    deflections = plane.deflections_from_grid(
        grid=grid, return_in_2d=False, return_binned=True
    )
    deflections_y = grid.mapping.scaled_array_2d_from_array_1d(
        array_1d=deflections[:, 0]
    )

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=plane,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=deflections_y,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        lines=lines,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=plane.kpc_per_arcsec,
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


def plot_deflections_x(
    plane,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Plane Deflections (x)",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="plane_deflections_x",
):

    deflections = plane.deflections_from_grid(
        grid=grid, return_in_2d=False, return_binned=True
    )
    deflections_x = grid.mapping.scaled_array_2d_from_array_1d(
        array_1d=deflections[:, 1]
    )

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=plane,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=deflections_x,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        lines=lines,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=plane.kpc_per_arcsec,
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


def plot_magnification(
    plane,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Plane Magnification",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="plane_magnification",
):

    magnification = plane.magnification_from_grid(
        grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    )

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=plane,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=magnification,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        lines=lines,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=plane.kpc_per_arcsec,
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


def plot_image_and_source_plane_subplot(
    image_plane,
    source_plane,
    grid,
    points=None,
    plot_critical_curves=False,
    plot_caustics=False,
    axis_limits=None,
    units="arcsec",
    output_path=None,
    output_format="show",
    output_filename="image_and_source_plane_grids",
):

    rows, columns, figsize = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=2
    )

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=image_plane, grid=grid, plot_critical_curve=True, plot_caustics=True
    )

    if plot_critical_curves:
        critical_curves = [lines[0]]
    else:
        critical_curves = None

    if plot_caustics:
        caustics = [lines[1]]
    else:
        caustics = None

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_plane_grid(
        plane=image_plane,
        grid=grid,
        axis_limits=axis_limits,
        points=points,
        lines=critical_curves,
        as_subplot=True,
        units=units,
        pointsize=3,
        xyticksize=16,
        titlesize=10,
        xlabelsize=10,
        ylabelsize=10,
        title="Image-plane Grid",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    source_plane_grid = image_plane.traced_grid_from_grid(grid=grid)

    plt.subplot(rows, columns, 2)

    plot_plane_grid(
        plane=source_plane,
        grid=source_plane_grid,
        axis_limits=axis_limits,
        points=points,
        lines=caustics,
        as_subplot=True,
        units=units,
        pointsize=3,
        xyticksize=16,
        titlesize=10,
        xlabelsize=10,
        ylabelsize=10,
        title="Source-plane Grid",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def plot_plane_grid(
    plane,
    grid,
    axis_limits=None,
    points=None,
    lines=None,
    as_subplot=False,
    units="arcsec",
    figsize=(12, 8),
    pointsize=3,
    title="Plane Grid",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    output_path=None,
    output_format="show",
    output_filename="plane_grid",
):

    grid_plotters.plot_grid(
        grid=grid,
        points=points,
        axis_limits=axis_limits,
        lines=lines,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=plane.kpc_per_arcsec,
        figsize=figsize,
        pointsize=pointsize,
        xyticksize=xyticksize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )
