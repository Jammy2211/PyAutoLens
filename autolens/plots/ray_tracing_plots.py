import autofit as af
import matplotlib
import numpy as np

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoarray.plotters import plotters, array_plotters
from autoastro.plots import lensing_plotters
from autolens.plots import plane_plots


def subplot(
    tracer,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
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

    array_plotter = array_plotter.plotter_as_sub_plotter()
    array_plotter = array_plotter.plotter_with_new_output_filename(
        output_filename="tracer"
    )

    rows, columns, figsize_tool = array_plotter.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if array_plotter.figsize is None:
        figsize = figsize_tool
    else:
        figsize = array_plotter.figsize

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    profile_image(
        tracer=tracer,
        grid=grid,
        mask=mask,
        positions=positions,
        include=include,
        array_plotter=array_plotter,
    )

    if tracer.has_mass_profile:

        plt.subplot(rows, columns, 2)

        convergence(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include=include,
            array_plotter=array_plotter,
        )

        plt.subplot(rows, columns, 3)

        potential(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include=include,
            array_plotter=array_plotter,
        )

    plt.subplot(rows, columns, 4)

    source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[-1]

    plane_plots.plane_image(
        plane=tracer.source_plane,
        grid=source_plane_grid,
        lines=include.caustics_from_obj(obj=tracer),
        include=include,
        array_plotter=array_plotter,
    )

    if tracer.has_mass_profile:

        plt.subplot(rows, columns, 5)

        deflections_y(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include=include,
            array_plotter=array_plotter,
        )

        plt.subplot(rows, columns, 6)

        deflections_x(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include=include,
            array_plotter=array_plotter,
        )

    array_plotter.output.to_figure(structure=None, is_sub_plotter=False)

    plt.close()


def individual(
    tracer,
    grid,
    mask=None,
    positions=None,
    plot_profile_image=False,
    plot_source_plane=False,
    plot_convergence=False,
    plot_potential=False,
    plot_deflections=False,
    include=lensing_plotters.Include(),
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
            positions=positions,
            include=include,
            array_plotter=array_plotter,
        )

    if plot_convergence:

        convergence(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include=include,
            array_plotter=array_plotter,
        )

    if plot_potential:

        potential(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include=include,
            array_plotter=array_plotter,
        )

    if plot_source_plane:

        source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[-1]

        plane_plots.plane_image(
            plane=tracer.source_plane,
            grid=source_plane_grid,
            lines=include.caustics_from_obj(obj=tracer),
            positions=None,
            include=include,
            array_plotter=array_plotter.plotter_with_new_output_filename(
                output_filename="source_plane"
            ),
        )

    if plot_deflections:

        deflections_y(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include=include,
            array_plotter=array_plotter,
        )

        deflections_x(
            tracer=tracer,
            grid=grid,
            mask=mask,
            include=include,
            array_plotter=array_plotter,
        )


@plotters.set_labels
def profile_image(
    tracer,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter.plot_array(
        array=tracer.profile_image_from_grid(grid=grid),
        mask=mask,
        points=positions,
        lines=include.critical_curves_from_obj(obj=tracer),
        centres=include.mass_profile_centres_of_planes_from_obj(obj=tracer),
    )


@plotters.set_labels
def convergence(
    tracer,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter.plot_array(
        array=tracer.convergence_from_grid(grid=grid),
        mask=mask,
        lines=include.critical_curves_from_obj(obj=tracer),
        centres=include.mass_profile_centres_of_planes_from_obj(obj=tracer),
    )


@plotters.set_labels
def potential(
    tracer,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter.plot_array(
        array=tracer.potential_from_grid(grid=grid),
        mask=mask,
        lines=include.critical_curves_from_obj(obj=tracer),
        centres=include.mass_profile_centres_of_planes_from_obj(obj=tracer),
    )


@plotters.set_labels
def deflections_y(
    tracer,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    deflections = tracer.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 0]
    )

    array_plotter.plot_array(
        array=deflections_y,
        mask=mask,
        lines=include.critical_curves_from_obj(obj=tracer),
        centres=include.mass_profile_centres_of_planes_from_obj(obj=tracer),
    )


@plotters.set_labels
def deflections_x(
    tracer,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    deflections = tracer.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 1]
    )

    array_plotter.plot_array(
        array=deflections_x,
        mask=mask,
        lines=include.critical_curves_from_obj(obj=tracer),
        centres=include.mass_profile_centres_of_planes_from_obj(obj=tracer),
    )

@plotters.set_labels
def contribution_map(
    tracer,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter.plot_array(
        array=tracer.contribution_map,
        mask=mask,
        points=positions,
        lines=include.critical_curves_from_obj(obj=tracer),
        centres=include.mass_profile_centres_of_planes_from_obj(obj=tracer),
    )
