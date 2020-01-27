from autoarray.plot import plotters
from autoastro.plot import lensing_plotters


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def profile_image(plane, grid, positions=None, include=None, plotter=None):

    plotter.plot_array(
        array=plane.profile_image_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=plane),
        light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def plane_image(plane, grid, positions=None, caustics=None, include=None, plotter=None):

    plotter.plot_array(
        array=plane.plane_image_from_grid(grid=grid).array,
        positions=positions,
        caustics=caustics,
        grid=include.grid_from_grid(grid=grid),
        light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def convergence(plane, grid, include=None, plotter=None):

    plotter.plot_array(
        array=plane.convergence_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        critical_curves=include.critical_curves_from_obj(obj=plane),
        light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def potential(plane, grid, include=None, plotter=None):

    plotter.plot_array(
        array=plane.potential_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        critical_curves=include.critical_curves_from_obj(obj=plane),
        light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def deflections_y(plane, grid, include=None, plotter=None):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 0]
    )

    plotter.plot_array(
        array=deflections_y,
        mask=include.mask_from_grid(grid=grid),
        critical_curves=include.critical_curves_from_obj(obj=plane),
        light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def deflections_x(plane, grid, include=None, plotter=None):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 1]
    )

    plotter.plot_array(
        array=deflections_x,
        mask=include.mask_from_grid(grid=grid),
        critical_curves=include.critical_curves_from_obj(obj=plane),
        light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def magnification(plane, grid, include=None, plotter=None):

    plotter.plot_array(
        array=plane.magnification_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        critical_curves=include.critical_curves_from_obj(obj=plane),
        light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_labels
def image_and_source_plane_subplot(
    image_plane,
    source_plane,
    grid,
    indexes=None,
    positions=None,
    axis_limits=None,
    include=None,
    sub_plotter=None,
):

    number_subplots = 2

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    plane_grid(
        plane=image_plane,
        grid=grid,
        indexes=indexes,
        axis_limits=axis_limits,
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=image_plane),
        include=include,
        plotter=sub_plotter,
    )

    source_plane_grid = image_plane.traced_grid_from_grid(grid=grid)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    plane_grid(
        plane=source_plane,
        grid=source_plane_grid,
        indexes=indexes,
        axis_limits=axis_limits,
        positions=positions,
        caustics=include.caustics_from_obj(obj=image_plane),
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.figure.close()


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def plane_grid(
    plane,
    grid,
    indexes=None,
    axis_limits=None,
    positions=None,
    critical_curves=None,
    caustics=None,
    include=None,
    plotter=None,
):

    plotter.plot_grid(
        grid=grid,
        positions=positions,
        axis_limits=axis_limits,
        indexes=indexes,
        critical_curves=critical_curves,
        caustics=caustics,
        light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        include_origin=include.origin,
        include_border=include.border,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def contribution_map(plane, mask=None, positions=None, include=None, plotter=None):

    plotter.plot_array(
        array=plane.contribution_map,
        mask=mask,
        positions=positions,
        light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
            obj=plane
        ),
        critical_curves=include.critical_curves_from_obj(obj=plane),
        include_origin=include.origin,
        include_border=include.border,
    )
