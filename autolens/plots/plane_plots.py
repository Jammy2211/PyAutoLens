import autofit as af
from autoarray.plotters import plotters, array_plotters, grid_plotters
import matplotlib

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoastro.plots import lensing_plotters


@plotters.set_labels
def profile_image(
    plane,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter.plot_array(
        array=plane.profile_image_from_grid(grid=grid),
        mask=mask,
        points=positions,
        grid=grid,
        lines=include.critical_curves_from_obj(obj=plane),
        centres=include.mass_profile_centres_of_galaxies_from_obj(obj=plane),
    )


@plotters.set_labels
def plane_image(
    plane,
    grid,
    positions=None,
    lines=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter.plot_array(
        array=plane.plane_image_from_grid(grid=grid).array,
        points=positions,
        grid=include.grid_from_grid(grid=grid),
        lines=include.critical_curves_from_obj(obj=plane),
        include_origin=include.origin,
    )


@plotters.set_labels
def convergence(
    plane,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter.plot_array(
        array=plane.convergence_from_grid(grid=grid),
        mask=mask,
        lines=include.critical_curves_from_obj(obj=plane),
        centres=include.mass_profile_centres_of_galaxies_from_obj(obj=plane),
        include_origin=include.origin,
    )


@plotters.set_labels
def potential(
    plane,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter.plot_array(
        array=plane.potential_from_grid(grid=grid),
        mask=mask,
        lines=include.critical_curves_from_obj(obj=plane),
        centres=include.mass_profile_centres_of_galaxies_from_obj(obj=plane),
        include_origin=include.origin,
    )


@plotters.set_labels
def deflections_y(
    plane,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 0]
    )

    array_plotter.plot_array(
        array=deflections_y,
        mask=mask,
        lines=include.critical_curves_from_obj(obj=plane),
        centres=include.mass_profile_centres_of_galaxies_from_obj(obj=plane),
        include_origin=include.origin,
    )


@plotters.set_labels
def deflections_x(
    plane,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 1]
    )

    array_plotter.plot_array(
        array=deflections_x,
        mask=mask,
        lines=include.critical_curves_from_obj(obj=plane),
        centres=include.mass_profile_centres_of_galaxies_from_obj(obj=plane),
        include_origin=include.origin,
    )


@plotters.set_labels
def magnification(
    plane,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter.plot_array(
        array=plane.magnification_from_grid(grid=grid),
        mask=mask,
        lines=include.critical_curves_from_obj(obj=plane),
        centres=include.mass_profile_centres_of_galaxies_from_obj(obj=plane),
        include_origin=include.origin,
    )


@plotters.set_labels
def image_and_source_plane_subplot(
    image_plane,
    source_plane,
    grid,
    points=None,
    axis_limits=None,
    include=lensing_plotters.Include(),
    grid_plotter=grid_plotters.GridPlotter(),
):

    rows, columns, figsize_tool = grid_plotter.get_subplot_rows_columns_figsize(
        number_subplots=2
    )

    if grid_plotter.figsize is None:
        figsize = figsize_tool
    else:
        figsize = grid_plotter.figsize

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plane_grid(
        plane=image_plane,
        grid=grid,
        axis_limits=axis_limits,
        points=points,
        lines=include.critical_curves_from_obj(obj=image_plane),
        include=include,
        grid_plotter=grid_plotter,
    )

    source_plane_grid = image_plane.traced_grid_from_grid(grid=grid)

    plt.subplot(rows, columns, 2)

    plane_grid(
        plane=source_plane,
        grid=source_plane_grid,
        axis_limits=axis_limits,
        points=points,
        lines=include.caustics_from_obj(obj=image_plane),
        include=include,
        grid_plotter=grid_plotter,
    )

    grid_plotter.output.to_figure(structure=None, is_sub_plotter=False)
    plt.close()


@plotters.set_labels
def plane_grid(
    plane,
    grid,
    axis_limits=None,
    points=None,
    lines=None,
    include=lensing_plotters.Include(),
    grid_plotter=grid_plotters.GridPlotter(),
):

    grid_plotter.plot_grid(
        grid=grid,
        points=points,
        axis_limits=axis_limits,
        lines=lines,
        centres=include.mass_profile_centres_of_galaxies_from_obj(obj=plane),
    )
