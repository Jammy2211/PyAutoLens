import autofit as af
from autoarray.plotters import plotters, array_plotters, grid_plotters
from autoarray.util import plotter_util
from autoastro.plots import lens_plotter_util
import matplotlib

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autoastro.plots import lens_plotter_util


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def profile_image(
    plane,
    grid,
    mask=None,
    positions=None,
    include_grid=False,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    profile_image = plane.profile_image_from_grid(grid=grid)

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    if not include_grid:
        grid = None

    array_plotter.plot_array(
        array=profile_image, mask=mask, points=positions, grid=grid, lines=lines
    )


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def plane_image(
    plane,
    grid,
    include_origin=None,
    positions=None,
    include_grid=False,
    lines=None,
    array_plotter=array_plotters.ArrayPlotter(),
):

    plane_image = plane.plane_image_from_grid(grid=grid)

    if include_grid:
        grid = plane_image.grid
    else:
        grid = None

    array_plotter.plot_array(
        array=plane_image.array,
        include_origin=include_origin,
        points=positions,
        grid=grid,
        lines=lines,
    )


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def convergence(
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    convergence = plane.convergence_from_grid(grid=grid)

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(array=convergence, mask=mask, lines=lines)


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def potential(
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    potential = plane.potential_from_grid(grid=grid)

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(array=potential, mask=mask, lines=lines)


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def deflections_y(
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 0]
    )

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(array=deflections_y, mask=mask, lines=lines)


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def deflections_x(
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 1]
    )

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(array=deflections_x, mask=mask, lines=lines)


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def magnification(
    plane,
    grid,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    magnification = plane.magnification_from_grid(grid=grid)

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=plane,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(array=magnification, mask=mask, lines=lines)


@plotters.set_includes
def image_and_source_plane_subplot(
    image_plane,
    source_plane,
    grid,
    points=None,
    include_critical_curves=False,
    include_caustics=False,
    axis_limits=None,
    grid_plotter=grid_plotters.GridPlotter(),
):

    rows, columns, figsize = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=2
    )

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
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
        grid_plotter=grid_plotter,
    )

    source_plane_grid = image_plane.traced_grid_from_grid(grid=grid)

    plt.subplot(rows, columns, 2)

    plane_grid(
        plane=source_plane,
        grid=source_plane_grid,
        axis_limits=axis_limits,
        points=points,
        lines=caustics,
        grid_plotter=grid_plotter,
    )

    grid_plotter.output_subplot_array()
    plt.close()


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def plane_grid(
    plane,
    grid,
    axis_limits=None,
    points=None,
    lines=None,
    grid_plotter=grid_plotters.GridPlotter(),
):

    grid_plotter.plot_grid(
        grid=grid, points=points, axis_limits=axis_limits, lines=lines
    )
