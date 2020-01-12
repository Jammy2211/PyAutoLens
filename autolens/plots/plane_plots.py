from autoarray.plotters import plotters
from autoastro.plots import lensing_plotters


@plotters.set_labels
def profile_image(
    plane,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
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
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
        array=plane.plane_image_from_grid(grid=grid).array,
        points=positions,
        grid=include.grid_from_grid(grid=grid),
        lines=lines,
        include_origin=include.origin,
    )


@plotters.set_labels
def convergence(
    plane,
    grid,
    mask=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
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
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
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
    plotter=plotters.Plotter(),
):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 0]
    )

    plotter.array.plot(
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
    plotter=plotters.Plotter(),
):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 1]
    )

    plotter.array.plot(
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
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
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
    sub_plotter=plotters.SubPlotter(),
):

    number_subplots = 2

    sub_plotter.setup_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    plane_grid(
        plane=image_plane,
        grid=grid,
        axis_limits=axis_limits,
        points=points,
        lines=include.critical_curves_from_obj(obj=image_plane),
        include=include,
        plotter=sub_plotter,
    )

    source_plane_grid = image_plane.traced_grid_from_grid(grid=grid)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 2)

    plane_grid(
        plane=source_plane,
        grid=source_plane_grid,
        axis_limits=axis_limits,
        points=points,
        lines=include.caustics_from_obj(obj=image_plane),
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.close_figure()


@plotters.set_labels
def plane_grid(
    plane,
    grid,
    axis_limits=None,
    points=None,
    lines=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
        grid=grid,
        points=points,
        axis_limits=axis_limits,
        lines=lines,
    )

@plotters.set_labels
def contribution_map(
    plane,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
        array=plane.contribution_map,
        mask=mask,
        points=positions,
        lines=include.critical_curves_from_obj(obj=plane),
        centres=include.mass_profile_centres_of_galaxies_from_obj(obj=plane),
    )

