from autoarray.plot import plotters
from autoastro.plot import lensing_plotters


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_inversion(
    inversion,
    image_positions=None,
    source_positions=None,
    grid=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    critical_curves=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    sub_plotter=None,
):

    number_subplots = 6

    ratio = float(
        (
            inversion.mapper.grid.scaled_maxima[1]
            - inversion.mapper.grid.scaled_minima[1]
        )
        / (
            inversion.mapper.grid.scaled_maxima[0]
            - inversion.mapper.grid.scaled_minima[0]
        )
    )

    if sub_plotter.figure.aspect in "square":
        aspect_inv = ratio
    elif sub_plotter.figure.aspect in "auto":
        aspect_inv = 1.0 / ratio
    elif sub_plotter.figure.aspect in "equal":
        aspect_inv = 1.0

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    reconstructed_image(
        inversion=inversion,
        image_positions=image_positions,
        grid=grid,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        critical_curves=critical_curves,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(
        number_subplots=number_subplots, subplot_index=2, aspect=aspect_inv
    )

    reconstruction(
        inversion=inversion,
        source_positions=source_positions,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(
        number_subplots=number_subplots, subplot_index=3, aspect=aspect_inv
    )

    errors(
        inversion=inversion,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        caustics=caustics,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(
        number_subplots=number_subplots, subplot_index=4, aspect=aspect_inv
    )

    residual_map(
        inversion=inversion,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        caustics=caustics,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(
        number_subplots=number_subplots, subplot_index=5, aspect=aspect_inv
    )

    chi_squared_map(
        inversion=inversion,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        caustics=caustics,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(
        number_subplots=number_subplots, subplot_index=6, aspect=aspect_inv
    )

    regularization_weights(
        inversion=inversion,
        source_positions=source_positions,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        caustics=caustics,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individuals(
    inversion,
    image_positions=None,
    source_positions=None,
    grid=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    critical_curves=None,
    caustics=None,
    plot_reconstructed_image=False,
    plot_reconstruction=False,
    plot_errors=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_regularization_weight_map=False,
    plot_interpolated_reconstruction=False,
    plot_interpolated_errors=False,
    include=None,
    plotter=None,
):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    if plot_reconstructed_image:

        reconstructed_image(
            inversion=inversion,
            image_positions=image_positions,
            grid=grid,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            critical_curves=critical_curves,
            include=include,
            plotter=plotter,
        )

    if plot_reconstruction:

        reconstruction(
            inversion=inversion,
            source_positions=source_positions,
            caustics=caustics,
            include=include,
            plotter=plotter,
        )

    if plot_errors:

        errors(inversion=inversion, include=include, caustics=caustics, plotter=plotter)

    if plot_residual_map:

        residual_map(
            inversion=inversion, include=include, caustics=caustics, plotter=plotter
        )

    if plot_normalized_residual_map:

        normalized_residual_map(
            inversion=inversion, caustics=caustics, include=include, plotter=plotter
        )

    if plot_chi_squared_map:

        chi_squared_map(
            inversion=inversion, caustics=caustics, include=include, plotter=plotter
        )

    if plot_regularization_weight_map:

        regularization_weights(
            inversion=inversion, caustics=caustics, include=include, plotter=plotter
        )

    if plot_interpolated_reconstruction:

        interpolated_reconstruction(
            inversion=inversion,
            source_positions=source_positions,
            caustics=caustics,
            include=include,
            plotter=plotter,
        )

    if plot_interpolated_errors:

        interpolated_errors(
            inversion=inversion, caustics=caustics, include=include, plotter=plotter
        )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def reconstructed_image(
    inversion,
    image_positions=None,
    grid=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    critical_curves=None,
    include=None,
    plotter=None,
):

    plotter.plot_array(
        array=inversion.mapped_reconstructed_image,
        mask=include.mask_from_grid(grid=inversion.mapper.grid),
        positions=image_positions,
        grid=grid,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        critical_curves=critical_curves,
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def reconstruction(
    inversion,
    source_positions=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.reconstruction
    )

    plotter.plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        positions=source_positions,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include.origin,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def errors(
    inversion,
    source_positions=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.errors
    )

    plotter.plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        positions=source_positions,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include.origin,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def residual_map(
    inversion,
    source_positions=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.residual_map
    )

    plotter.plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        positions=source_positions,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include.origin,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def normalized_residual_map(
    inversion,
    source_positions=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.normalized_residual_map
    )

    plotter.plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        positions=source_positions,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include.origin,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def chi_squared_map(
    inversion,
    source_positions=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):

    source_pixel_values = inversion.mapper.reconstructed_pixelization_from_solution_vector(
        solution_vector=inversion.chi_squared_map
    )

    plotter.plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=source_pixel_values,
        positions=source_positions,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include.origin,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def regularization_weights(
    inversion,
    source_positions=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):

    regularization_weights = inversion.regularization.regularization_weights_from_mapper(
        mapper=inversion.mapper
    )

    plotter.plot_mapper(
        mapper=inversion.mapper,
        source_pixel_values=regularization_weights,
        positions=source_positions,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include.origin,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def interpolated_reconstruction(
    inversion,
    source_positions=None,
    grid=None,
    caustics=None,
    include=None,
    plotter=None,
):

    plotter.plot_array(
        array=inversion.interpolated_reconstruction_from_shape_2d(),
        positions=source_positions,
        grid=grid,
        caustics=caustics,
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def interpolated_errors(
    inversion,
    source_positions=None,
    grid=None,
    caustics=None,
    include=None,
    plotter=None,
):

    plotter.plot_array(
        array=inversion.interpolated_errors_from_shape_2d(),
        positions=source_positions,
        grid=grid,
        critical_curves=caustics,
        include_origin=include.origin,
        include_border=include.border,
    )
