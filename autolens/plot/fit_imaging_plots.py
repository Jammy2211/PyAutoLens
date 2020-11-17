import numpy as np
from autoarray.plot import plotters
from autogalaxy.plot import lensing_plotters, plane_plots, inversion_plots


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_fit_imaging(fit, include=None, sub_plotter=None):
    number_subplots = 6

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    image(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    signal_to_noise_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    model_image(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    residual_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    normalized_residual_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    chi_squared_map(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def subplots_of_all_planes(fit, include=None, sub_plotter=None):

    for plane_index in range(fit.tracer.total_planes):

        if (
            fit.tracer.planes[plane_index].has_light_profile
            or fit.tracer.planes[plane_index].has_pixelization
        ):

            subplot_of_plane(
                fit=fit,
                plane_index=plane_index,
                include=include,
                sub_plotter=sub_plotter,
            )


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_of_plane(fit, plane_index, include=None, sub_plotter=None):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    number_subplots = 4

    sub_plotter = sub_plotter.plotter_with_new_output(
        filename=sub_plotter.output.filename + "_" + str(plane_index)
    )

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    image(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    subtracted_image_of_plane(
        fit=fit, plane_index=plane_index, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    model_image_of_plane(
        fit=fit, plane_index=plane_index, include=include, plotter=sub_plotter
    )

    if not fit.tracer.planes[plane_index].has_pixelization:

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

        traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

        plane_plots.plane_image(
            plane=fit.tracer.planes[plane_index],
            grid=traced_grids[plane_index],
            positions=include.positions_of_plane_from_fit_and_plane_index(
                fit=fit, plane_index=plane_index
            ),
            caustics=include.caustics_from_obj(obj=fit.tracer),
            include=include,
            plotter=sub_plotter,
        )

    elif fit.tracer.planes[plane_index].has_pixelization:

        ratio = float(
            (
                fit.inversion.mapper.grid.scaled_maxima[1]
                - fit.inversion.mapper.grid.scaled_minima[1]
            )
            / (
                fit.inversion.mapper.grid.scaled_maxima[0]
                - fit.inversion.mapper.grid.scaled_minima[0]
            )
        )

        if sub_plotter.figure.aspect in "square":
            aspect_inv = ratio
        elif sub_plotter.figure.aspect in "auto":
            aspect_inv = 1.0 / ratio
        elif sub_plotter.figure.aspect in "equal":
            aspect_inv = 1.0

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=4, aspect=float(aspect_inv)
        )

        inversion_plots.reconstruction(
            inversion=fit.inversion,
            source_positions=include.positions_of_plane_from_fit_and_plane_index(
                fit=fit, plane_index=plane_index
            ),
            caustics=include.caustics_from_obj(obj=fit.tracer),
            include=include,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individuals(
    fit,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_image=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_subtracted_images_of_planes=False,
    plot_model_images_of_planes=False,
    plot_plane_images_of_planes=False,
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

    if plot_image:

        image(fit=fit, include=include, plotter=plotter)

    if plot_noise_map:

        noise_map(fit=fit, include=include, plotter=plotter)

    if plot_signal_to_noise_map:

        signal_to_noise_map(fit=fit, include=include, plotter=plotter)

    if plot_model_image:

        model_image(fit=fit, include=include, plotter=plotter)

    if plot_residual_map:

        residual_map(fit=fit, include=include, plotter=plotter)

    if plot_normalized_residual_map:

        normalized_residual_map(fit=fit, include=include, plotter=plotter)

    if plot_chi_squared_map:

        chi_squared_map(fit=fit, include=include, plotter=plotter)

    if plot_subtracted_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            subtracted_image_of_plane(
                fit=fit, plane_index=plane_index, include=include, plotter=plotter
            )

    if plot_model_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            model_image_of_plane(
                fit=fit, plane_index=plane_index, include=include, plotter=plotter
            )

    if plot_plane_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            plotter = plotter.plotter_with_new_output(
                filename="plane_image_of_plane_" + str(plane_index)
            )

            if fit.tracer.planes[plane_index].has_light_profile:

                plane_plots.plane_image(
                    plane=fit.tracer.planes[plane_index],
                    grid=include.traced_grid_of_plane_from_fit_and_plane_index(
                        fit=fit, plane_index=plane_index
                    ),
                    positions=include.positions_of_plane_from_fit_and_plane_index(
                        fit=fit, plane_index=plane_index
                    ),
                    caustics=include.caustics_from_obj(obj=fit.tracer),
                    include=include,
                    plotter=plotter,
                )

            elif fit.tracer.planes[plane_index].has_pixelization:

                inversion_plots.reconstruction(
                    inversion=fit.inversion,
                    source_positions=include.positions_of_plane_from_fit_and_plane_index(
                        fit=fit, plane_index=plane_index
                    ),
                    caustics=include.caustics_from_obj(obj=fit.tracer),
                    include=include,
                    plotter=plotter,
                )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def subtracted_image_of_plane(fit, plane_index, include=None, plotter=None):
    """Plot the model image of a specific plane of a lens fit.

    Set *autolens.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    plane_indexes : int
        The plane from which the model image is generated.
    """

    if isinstance(plotter, lensing_plotters.Plotter):
        plotter = plotter.plotter_with_new_output(
            filename=plotter.output.filename + "_" + str(plane_index)
        )

    if fit.tracer.total_planes > 1:

        other_planes_model_images = [
            model_image
            for i, model_image in enumerate(fit.model_images_of_planes)
            if i != plane_index
        ]

        subtracted_image = fit.image - sum(other_planes_model_images)

    else:

        subtracted_image = fit.image

    plotter_norm = plotter.plotter_with_new_cmap(
        norm_max=np.max(fit.model_images_of_planes[plane_index]),
        norm_min=np.min(fit.model_images_of_planes[plane_index]),
    )

    plotter_norm.plot_array(
        array=subtracted_image,
        mask=include.mask_from_fit(fit=fit),
        grid=include.inversion_image_pixelization_grid_from_fit(fit=fit),
        positions=include.positions_from_fit(fit=fit),
        critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
        light_profile_centres=include.light_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        mass_profile_centres=include.mass_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def model_image_of_plane(fit, plane_index, include=None, plotter=None):
    """Plot the model image of a specific plane of a lens fit.

    Set *autolens.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    plane_indexes : [int]
        The plane from which the model image is generated.
    """

    if isinstance(plotter, lensing_plotters.Plotter):
        plotter = plotter.plotter_with_new_output(
            filename=plotter.output.filename + "_" + str(plane_index)
        )

    plotter.plot_array(
        array=fit.model_images_of_planes[plane_index],
        mask=include.mask_from_fit(fit=fit),
        positions=include.positions_from_fit(fit=fit),
        critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
        light_profile_centres=include.light_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        mass_profile_centres=include.mass_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def image(fit, array_overlay=None, include=None, plotter=None):
    """Plot the image of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    plotter.plot_array(
        array=fit.data,
        mask=include.mask_from_fit(fit=fit),
        grid=include.inversion_image_pixelization_grid_from_fit(fit=fit),
        positions=include.positions_from_fit(fit=fit),
        light_profile_centres=include.light_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        mass_profile_centres=include.mass_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
        include_origin=include.origin,
        array_overlay=array_overlay,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def noise_map(fit, include=None, plotter=None):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map, PSF, signal-to-noise_map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter.plot_array(
        array=fit.noise_map,
        mask=include.mask_from_fit(fit=fit),
        light_profile_centres=include.light_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        mass_profile_centres=include.mass_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def signal_to_noise_map(fit, include=None, plotter=None):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.imaging.datas.Imaging
    The datas-datas, which include the observed datas, signal_to_noise_map, PSF, signal-to-signal_to_noise_map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter.plot_array(
        array=fit.signal_to_noise_map,
        mask=include.mask_from_fit(fit=fit),
        light_profile_centres=include.light_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        mass_profile_centres=include.mass_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def model_image(fit, include=None, plotter=None):
    """Plot the model image of a fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    """
    plotter.plot_array(
        array=fit.model_data,
        mask=include.mask_from_fit(fit=fit),
        positions=include.positions_from_fit(fit=fit),
        light_profile_centres=include.light_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        mass_profile_centres=include.mass_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def residual_map(fit, include=None, plotter=None):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """
    plotter.plot_array(
        array=fit.residual_map,
        mask=include.mask_from_fit(fit=fit),
        light_profile_centres=include.light_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        mass_profile_centres=include.mass_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def normalized_residual_map(fit, include=None, plotter=None):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, normalized_residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the normalized_residual_map are plotted.
    """
    plotter.plot_array(
        array=fit.normalized_residual_map,
        mask=include.mask_from_fit(fit=fit),
        light_profile_centres=include.light_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        mass_profile_centres=include.mass_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def chi_squared_map(fit, include=None, plotter=None):
    """Plot the chi-squared-map of a lens fit.

    Set *autolens.datas.array.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the chi-squareds are plotted.
    """
    plotter.plot_array(
        array=fit.chi_squared_map,
        mask=include.mask_from_fit(fit=fit),
        light_profile_centres=include.light_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        mass_profile_centres=include.mass_profile_centres_from_obj(
            obj=fit.tracer.image_plane
        ),
        critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
        include_origin=include.origin,
    )
