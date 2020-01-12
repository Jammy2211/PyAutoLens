import autoarray as aa
from autoarray.plotters import plotters, mat_objs
from autoastro.plots import lensing_plotters
from autolens.plots import plane_plots

@plotters.set_subplot_filename
def subplot_fit_imaging(
    fit,
    include=lensing_plotters.Include(),
    sub_plotter=plotters.SubPlotter(),
):

    aa.plot.fit_imaging.subplot_fit_imaging(
        fit=fit,
        grid=include.inversion_image_pixelization_grid_from_fit(fit=fit),
        points=include.positions_from_fit(fit=fit),
        lines=include.critical_curves_from_obj(obj=fit.tracer),
        include=include,
        sub_plotter=sub_plotter,
    )


def subplot_of_planes(
    fit,
    include=lensing_plotters.Include(),
    sub_plotter=plotters.SubPlotter(),
):

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

@plotters.set_subplot_filename
def subplot_of_plane(
    fit,
    plane_index,
    include=lensing_plotters.Include(),
    sub_plotter=plotters.SubPlotter(),
):
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
        output=mat_objs.Output(filename=sub_plotter.output.filename + "_" + str(plane_index))
    )

    sub_plotter.setup_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    aa.plot.fit_imaging.image(
        fit=fit,
        grid=include.inversion_image_pixelization_grid_from_fit(fit=fit),
        points=include.positions_from_fit(fit=fit),
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 2)

    subtracted_image_of_plane(
        fit=fit,
        plane_index=plane_index,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 3)

    model_image_of_plane(
        fit=fit,
        plane_index=plane_index,
        include=include,
        plotter=sub_plotter,
    )

    if not fit.tracer.planes[plane_index].has_pixelization:

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 4)

        traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

        plane_plots.plane_image(
            plane=fit.tracer.planes[plane_index],
            grid=traced_grids[plane_index],
            lines=include.caustics_from_obj(obj=fit.tracer),
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

        if sub_plotter.aspect is "square":
            aspect_inv = ratio
        elif sub_plotter.aspect is "auto":
            aspect_inv = 1.0 / ratio
        elif sub_plotter.aspect is "equal":
            aspect_inv = 1.0

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 4, aspect=float(aspect_inv))

        aa.plot.inversion.reconstruction(
            inversion=fit.inversion,
            lines=include.caustics_from_obj(obj=fit.tracer),
            include=include,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.close_figure()


def individuals(
    fit,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_image=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_inversion_reconstruction=False,
    plot_inversion_errors=False,
    plot_inversion_residual_map=False,
    plot_inversion_normalized_residual_map=False,
    plot_inversion_chi_squared_map=False,
    plot_inversion_regularization_weight_map=False,
    plot_inversion_interpolated_reconstruction=False,
    plot_inversion_interpolated_errors=False,
    plot_subtracted_images_of_planes=False,
    plot_model_images_of_planes=False,
    plot_plane_images_of_planes=False,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
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

    aa.plot.fit_imaging.individuals(
        fit=fit,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_model_image=plot_model_image,
        plot_residual_map=plot_residual_map,
        plot_normalized_residual_map=plot_normalized_residual_map,
        plot_chi_squared_map=plot_chi_squared_map,
        plot_inversion_reconstruction=plot_inversion_reconstruction,
        plot_inversion_errors=plot_inversion_errors,
        plot_inversion_residual_map=plot_inversion_residual_map,
        plot_inversion_normalized_residual_map=plot_inversion_normalized_residual_map,
        plot_inversion_chi_squared_map=plot_inversion_chi_squared_map,
        plot_inversion_regularization_weight_map=plot_inversion_regularization_weight_map,
        plot_inversion_interpolated_reconstruction=plot_inversion_interpolated_reconstruction,
        plot_inversion_interpolated_errors=plot_inversion_interpolated_errors,
        include=include,
        plotter=plotter,
    )

    traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

    if plot_subtracted_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            subtracted_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                include=include,
                plotter=plotter,
            )

    if plot_model_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            model_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                include=include,
                plotter=plotter,
            )

    if plot_plane_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            plotter = plotter.plotter_with_new_output(
                output=mat_objs.Output(filename="plane_image_of_plane_" + str(plane_index))
            )

            if fit.tracer.planes[plane_index].has_light_profile:

                plane_plots.plane_image(
                    plane=fit.tracer.planes[plane_index],
                    grid=traced_grids[plane_index],
                    lines=include.caustics_from_obj(obj=fit.tracer),
                    include=include,
                    plotter=plotter,
                )

            elif fit.tracer.planes[plane_index].has_pixelization:

                aa.plot.inversion.reconstruction(
                    inversion=fit.inversion,
                    lines=include.caustics_from_obj(obj=fit.tracer),
                    include=include,
                    plotter=plotter,
                )


@plotters.set_labels
def subtracted_image_of_plane(
    fit,
    plane_index,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
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

    plotter = plotter.plotter_with_new_output(
        output=mat_objs.Output(filename=plotter.output.filename + "_" + str(plane_index))
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

    plotter.array.plot(
        array=subtracted_image,
        mask=include.mask_from_fit(fit=fit),
        grid=include.inversion_image_pixelization_grid_from_fit(fit=fit),
        points=include.positions_from_fit(fit=fit),
        lines=include.critical_curves_from_obj(obj=fit.tracer),
        centres=include.mass_profile_centres_of_planes_from_obj(obj=fit.tracer),
    )


@plotters.set_labels
def model_image_of_plane(
    fit,
    plane_index,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the model image of a specific plane of a lens fit.

    Set *autolens.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    plane_indexes : [int]
        The plane from which the model image is generated.
    """

    plotter = plotter.plotter_with_new_output(
        output=mat_objs.Output(filename=plotter.output.filename + "_" + str(plane_index))
    )

    plotter.array.plot(
        array=fit.model_images_of_planes[plane_index],
        mask=include.mask_from_fit(fit=fit),
        lines=include.critical_curves_from_obj(obj=fit.tracer),
        points=include.positions_from_fit(fit=fit),
        centres=include.mass_profile_centres_of_planes_from_obj(obj=fit.tracer),
    )