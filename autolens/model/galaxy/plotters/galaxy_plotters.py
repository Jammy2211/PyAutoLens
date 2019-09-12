import autofit as af
import matplotlib

backend = af.conf.instance.visualize.get("figures", "backend", str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autolens.plotters import plotter_util, array_plotters
from autolens.model.profiles.plotters import profile_plotters


def plot_profile_image(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Galaxy Profile Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_profile_image",
):
    """Plot the image (e.g. the datas) of a galaxy, on a grid of (y,x) coordinates.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    galaxy : model.galaxy.galaxy.Galaxy
        The galaxy whose image are plotted.
    grid : ndarray or datas.array.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    image = galaxy.profile_image_from_grid(
        grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    )

    if galaxy.has_mass_profile:

        lines = plotter_util.get_critical_curve_and_caustic(
            obj=galaxy,
            grid=grid,
            plot_critical_curve=plot_critical_curves,
            plot_caustics=plot_caustics,
        )
    else:
        lines = None

    array_plotters.plot_array(
        array=image,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        lines=lines,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_convergence(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Galaxy Convergence",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_convergence",
):
    """Plot the convergence of a galaxy, on a grid of (y,x) coordinates.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    galaxy : model.galaxy.galaxy.Galaxy
        The galaxy whose convergence is plotted.
    grid : ndarray or datas.array.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    convergence = galaxy.convergence_from_grid(
        grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    )

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=galaxy,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=convergence,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        lines=lines,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_potential(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Galaxy Potential",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_potential",
):
    """Plot the potential of a galaxy, on a grid of (y,x) coordinates.

     Set *autolens.datas.array.plotters.array_plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
    galaxy : model.galaxy.galaxy.Galaxy
         The galaxy whose potential is plotted.
    grid : ndarray or datas.array.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
     """
    potential = galaxy.potential_from_grid(
        grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    )

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=galaxy,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=potential,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        lines=lines,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_deflections_y(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Galaxy Deflections (y)",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_deflections_y",
):
    """Plot the y component of the deflection angles of a galaxy, on a grid of (y,x) coordinates.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    galaxy : model.galaxy.galaxy.Galaxy
        The galaxy whose y deflecton angles are plotted.
    grid : ndarray or datas.array.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    deflections = galaxy.deflections_from_grid(
        grid, return_in_2d=False, return_binned=True
    )
    deflections_y = grid.mapping.scaled_array_2d_from_array_1d(deflections[:, 0])

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=galaxy,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=deflections_y,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        lines=lines,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_deflections_x(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Galaxy Deflections (x)",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_deflections_x",
):
    """Plot the x component of the deflection angles of a galaxy, on a grid of (y,x) coordinates.

     Set *autolens.datas.array.plotters.array_plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
    galaxy : model.galaxy.galaxy.Galaxy
         The galaxy whose x deflecton angles are plotted.
     grid : ndarray or datas.array.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
     """
    deflections = galaxy.deflections_from_grid(
        grid, return_in_2d=False, return_binned=True
    )
    deflections_x = grid.mapping.scaled_array_2d_from_array_1d(deflections[:, 1])

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=galaxy,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=deflections_x,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        lines=lines,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_magnification(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
    plot_critical_curves=False,
    plot_caustics=False,
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
    title="Galaxy Magnification",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_magnification",
):
    """Plot the magnification of a galaxy, on a grid of (y,x) coordinates.

     Set *autolens.datas.array.plotters.array_plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
    galaxy : model.galaxy.galaxy.Galaxy
         The galaxy whose magnification is plotted.
    grid : ndarray or datas.array.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
     """
    magnification = galaxy.magnification_from_grid(
        grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    )

    lines = plotter_util.get_critical_curve_and_caustic(
        obj=galaxy,
        grid=grid,
        plot_critical_curve=plot_critical_curves,
        plot_caustics=plot_caustics,
    )

    array_plotters.plot_array(
        array=magnification,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        lines=lines,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        grid_pointsize=grid_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def plot_image_subplot(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
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
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_image",
):

    total_light_profiles = len(galaxy.light_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_light_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, light_profile in enumerate(galaxy.light_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.plot_image(
            light_profile=light_profile,
            mask=mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            positions=positions,
            grid=grid,
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
            title="Galaxy Component",
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

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def plot_convergence_subplot(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
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
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_convergence",
):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_mass_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.plot_convergence(
            mass_profile=mass_profile,
            grid=grid,
            mask=mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            positions=positions,
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
            title="Galaxy Component",
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

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def plot_potential_subplot(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
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
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_potential",
):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_mass_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.plot_potential(
            mass_profile=mass_profile,
            grid=grid,
            mask=mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            positions=positions,
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
            title="Galaxy Component",
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

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def plot_deflections_y_subplot(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
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
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_deflections_y",
):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_mass_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.plot_deflections_y(
            mass_profile=mass_profile,
            grid=grid,
            mask=mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            positions=positions,
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
            title="Galaxy Component",
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

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def plot_deflections_x_subplot(
    galaxy,
    grid,
    mask=None,
    extract_array_from_mask=False,
    zoom_around_mask=False,
    positions=None,
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
    mask_pointsize=10,
    position_pointsize=10.0,
    grid_pointsize=1,
    output_path=None,
    output_format="show",
    output_filename="galaxy_individual_deflections_x",
):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=total_mass_profiles
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.plot_deflections_x(
            mass_profile=mass_profile,
            grid=grid,
            mask=mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            positions=positions,
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
            title="Galaxy Component",
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

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()
