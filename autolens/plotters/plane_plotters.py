import autofit as af
from autoarray.util import plotter_util
from autoastro.plotters import lens_plotter_util
import matplotlib

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

import autoarray as aa


def profile_image(
    plane,
    grid,
    mask=None,
    positions=None,
    include_grid=False,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
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
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="plane_profile_image",
):

    profile_image = plane.profile_image_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    if not include_grid:
        grid = None

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=profile_image,
        mask=mask,
        points=positions,
        grid=grid,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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
        point_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plane_image(
    plane,
    grid,
    include_origin=True,
    positions=None,
    include_grid=False,
    lines=None,
    as_subplot=False,
    plot_in_kpc=False,
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

    if include_grid:
        grid = plane_image.grid
    else:
        grid = None

    if include_origin:
        origin = plane_image.grid.origin
    else:
        origin = None

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=plane_image.array,
        include_origin=origin,
        points=positions,
        grid=grid,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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
        point_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def convergence(
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
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

    convergence = plane.convergence_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=convergence,
        mask=mask,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
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

    potential = plane.potential_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=potential,
        mask=mask,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
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

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 0]
    )

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=deflections_y,
        mask=mask,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
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

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 1]
    )

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=deflections_x,
        mask=mask,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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


def magnification(
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    as_subplot=False,
    plot_in_kpc=False,
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

    magnification = plane.magnification_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=magnification,
        mask=mask,
        lines=lines,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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


def image_and_source_plane_subplot(
    image_plane,
    source_plane,
    grid,
    points=None,
    include_critical_curves=False,
    include_caustics=False,
    axis_limits=None,
    plot_in_kpc=False,
    output_path=None,
    output_format="show",
    output_filename="image_and_source_plane_grids",
):

    rows, columns, figsize = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=2
    )

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=image_plane, include_critical_curves=True, include_caustics=True
    )

    if include_critical_curves:
        critical_curves = [lines[0]]
    else:
        critical_curves = None

    if include_caustics:
        caustics = [lines[1]]
    else:
        caustics = None

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plane_grid(
        plane=image_plane,
        grid=grid,
        axis_limits=axis_limits,
        points=points,
        lines=critical_curves,
        plot_in_kpc=plot_in_kpc,
        as_subplot=True,
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

    plane_grid(
        plane=source_plane,
        grid=source_plane_grid,
        axis_limits=axis_limits,
        points=points,
        lines=caustics,
        plot_in_kpc=plot_in_kpc,
        as_subplot=True,
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


def plane_grid(
    plane,
    grid,
    axis_limits=None,
    points=None,
    lines=None,
    as_subplot=False,
    plot_in_kpc=False,
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

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.grid(
        grid=grid,
        points=points,
        axis_limits=axis_limits,
        lines=lines,
        as_subplot=as_subplot,
        unit_label_y=unit_label,
        unit_conversion_factor=unit_conversion_factor,
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
