import autoarray as aa
from autoarray.plot import plotters
from autoastro.plot import lensing_plotters
from autolens.plot import plane_plots, ray_tracing_plots


def subplot_fit_interferometer(
    fit,
    include=lensing_plotters.Include(),
    sub_plotter=plotters.SubPlotter(),
):

    aa.plot.fit_interferometer.subplot_fit_interferometer(
        fit=fit,
        include=include,
        sub_plotter=sub_plotter,
    )

@plotters.set_subplot_filename
def subplot_fit_real_space(
    fit,
    include=lensing_plotters.Include(),
    sub_plotter=plotters.SubPlotter(),
):

    number_subplots = 2

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    if fit.inversion is None:

        ray_tracing_plots.profile_image(
            tracer=fit.tracer,
            grid=fit.masked_interferometer.grid,
            mask=include.real_space_mask_from_fit(fit=fit),
            positions=include.positions_from_fit(fit=fit),
            include=include,
            plotter=sub_plotter
        )

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 2)

        plane_plots.plane_image(
            plane=fit.tracer.source_plane,
            grid=fit.masked_interferometer.grid,
            caustics=include.caustics_from_obj(obj=fit.tracer),
            plotter=sub_plotter
        )

    elif fit.inversion is not None:

        aa.plot.inversion.reconstructed_image(
            inversion=fit.inversion,
            mask=include.real_space_mask_from_fit(fit=fit),
            lines=include.critical_curves_from_obj(obj=fit.tracer),
            positions=include.positions_from_fit(fit=fit),
            grid=include.inversion_image_pixelization_grid_from_fit(fit=fit),
            plotter=sub_plotter,
        )

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

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 2, aspect=float(aspect_inv))

        aa.plot.inversion.reconstruction(
            inversion=fit.inversion,
            lines=include.caustics_from_obj(obj=fit.tracer),
            include=include,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()

def individuals(
    fit,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_visibilities=False,
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
    include=lensing_plotters.Include(),
    plotter=lensing_plotters.Plotter(),
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

    aa.plot.fit_interferometer.individuals(
        fit=fit,
        plot_visibilities=plot_visibilities,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_model_visibilities=plot_model_visibilities,
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
