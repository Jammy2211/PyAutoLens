import autofit as af
import matplotlib
import numpy as np

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import plotters, array_plotters, grid_plotters
from autoarray.util import plotter_util
from autoastro.plots import lens_plotter_util
from autolens.plots import plane_plots


def subplot(
    tracer,
    grid,
    mask=None,
    include_multiple_images=False,
    include_mass_profile_centres=False,
    include_critical_curves=False,
    include_caustics=False,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
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
        include_multiple_images=include_multiple_images,
        include_mass_profile_centres=include_mass_profile_centres,
        include_critical_curves=include_critical_curves,
        positions=positions,
    )

    if tracer.has_mass_profile:

        plt.subplot(rows, columns, 2)

        convergence(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include_multiple_images=include_multiple_images,
            include_mass_profile_centres=include_mass_profile_centres,
        )

        plt.subplot(rows, columns, 3)

        potential(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include_multiple_images=include_multiple_images,
            include_mass_profile_centres=include_mass_profile_centres,
        )

    plt.subplot(rows, columns, 4)

    source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[-1]

    caustics = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=tracer, include_critical_curves=False, include_caustics=include_caustics
    )

    plane_plots.plane_image(
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
            include_multiple_images=include_multiple_images,
            include_mass_profile_centres=include_mass_profile_centres,
            as_subplot=True,
            plot_in_kpc=plot_in_kpc,
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
            include_multiple_images=include_multiple_images,
            include_mass_profile_centres=include_mass_profile_centres,
            as_subplot=True,
            plot_in_kpc=plot_in_kpc,
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
    include_multiple_images=False,
    include_mass_profile_centres=False,
    include_critical_curves=False,
    include_caustics=False,
    positions=None,
    plot_profile_image=False,
    plot_source_plane=False,
    plot_convergence=False,
    plot_potential=False,
    plot_deflections=False,
    array_plotter=array_plotters.ArrayPlotter(),
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
            include_multiple_images=include_multiple_images,
            include_mass_profile_centres=include_mass_profile_centres,
            include_critical_curves=include_critical_curves,
            positions=positions,
            array_plotter=array_plotter
        )

    if plot_convergence:

        convergence(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include_multiple_images=include_multiple_images,
            include_mass_profile_centres=include_mass_profile_centres,
            array_plotter=array_plotter
        )

    if plot_potential:

        potential(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include_multiple_images=include_multiple_images,
            include_mass_profile_centres=include_mass_profile_centres,
            array_plotter=array_plotter
        )

    if plot_source_plane:

        source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[-1]

        caustics = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
            obj=tracer, include_critical_curves=False, include_caustics=include_caustics
        )

        plane_plots.plane_image(
            plane=tracer.source_plane,
            grid=source_plane_grid,
            lines=caustics,
            positions=None,
            include_grid=False,
            array_plotter=array_plotter
        )

    if plot_deflections:

        deflections_y(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include_multiple_images=include_multiple_images,
            include_mass_profile_centres=include_mass_profile_centres,
            array_plotter=array_plotter
        )

        deflections_x(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include_multiple_images=include_multiple_images,
            include_mass_profile_centres=include_mass_profile_centres,
            array_plotter=array_plotter
        )


def profile_image(
    tracer,
    grid,
    mask=None,
    include_multiple_images=False,
    include_mass_profile_centres=False,
    include_critical_curves=False,
    include_caustics=False,
    positions=None,
    array_plotter=array_plotters.ArrayPlotter(),
):

    profile_image = tracer.profile_image_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    if include_multiple_images:
        positions = tracer.image_plane_multiple_image_positions_of_galaxies(grid=grid)[
            0
        ]
    else:
        multiple_images = None

    if include_mass_profile_centres:
        mass_profile_centres = tracer.image_plane.mass_profile_centres_of_galaxies
    else:
        mass_profile_centres = None

    array_plotter.plot_array(
        array=profile_image,
        mask=mask,
        lines=lines,
        points=positions,
        centres=mass_profile_centres,
    )


def convergence(
    tracer,
    grid,
    mask=None,
    include_multiple_images=False,
    include_mass_profile_centres=False,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    convergence = tracer.convergence_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    if include_mass_profile_centres:
        mass_profile_centres = tracer.image_plane.mass_profile_centres_of_galaxies
    else:
        mass_profile_centres = None

    array_plotter.plot_array(
        array=convergence,
        mask=mask,
        lines=lines,
        centres=mass_profile_centres,
    )


def potential(
    tracer,
    grid,
    mask=None,
    include_multiple_images=False,
    include_mass_profile_centres=False,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    potential = tracer.potential_from_grid(grid=grid)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    if include_mass_profile_centres:
        mass_profile_centres = tracer.image_plane.mass_profile_centres_of_galaxies
    else:
        mass_profile_centres = None

    array_plotter.plot_array(
        array=potential,
        mask=mask,
        lines=lines,
        centres=mass_profile_centres,
    )


def deflections_y(
    tracer,
    grid,
    mask=None,
    include_multiple_images=False,
    include_mass_profile_centres=False,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    deflections = tracer.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 0]
    )

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    if include_mass_profile_centres:
        mass_profile_centres = tracer.image_plane.mass_profile_centres_of_galaxies
    else:
        mass_profile_centres = None

    array_plotter.plot_array(
        array=deflections_y,
        mask=mask,
        lines=lines,
        centres=mass_profile_centres,
    )


def deflections_x(
    tracer,
    grid,
    mask=None,
    include_multiple_images=False,
    include_mass_profile_centres=False,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    deflections = tracer.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 1]
    )

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    if include_mass_profile_centres:
        mass_profile_centres = tracer.image_plane.mass_profile_centres_of_galaxies
    else:
        mass_profile_centres = None

    array_plotter.plot_array(
        array=deflections_x,
        lines=lines,
        mask=mask,
        centres=mass_profile_centres,
    )
