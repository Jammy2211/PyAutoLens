import autogalaxy as ag
from autogalaxy.plot.fit_interferometer_plots import *
from autolens.plot import ray_tracing_plots


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_fit_interferometer(fit, include=None, sub_plotter=None):

    number_subplots = 6

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    ag.plot.FitInterferometer.residual_map_vs_uv_distances(
        fit=fit, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    ag.plot.FitInterferometer.normalized_residual_map_vs_uv_distances(
        fit=fit, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    ag.plot.FitInterferometer.chi_squared_map_vs_uv_distances(
        fit=fit, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    ag.plot.FitInterferometer.residual_map_vs_uv_distances(
        fit=fit, plot_real=False, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    ag.plot.FitInterferometer.normalized_residual_map_vs_uv_distances(
        fit=fit, plot_real=False, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    ag.plot.FitInterferometer.chi_squared_map_vs_uv_distances(
        fit=fit, plot_real=False, include=include, plotter=sub_plotter
    )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_fit_real_space(fit, include=None, sub_plotter=None):

    number_subplots = 2

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    if fit.inversion is None:

        ray_tracing_plots.image(
            tracer=fit.tracer,
            grid=fit.masked_interferometer.grid,
            positions=include.positions_from_fit(fit=fit),
            include=include,
            plotter=sub_plotter,
        )

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        ag.plot.Plane.plane_image(
            plane=fit.tracer.source_plane,
            grid=fit.masked_interferometer.grid,
            positions=include.positions_of_plane_from_fit_and_plane_index(
                fit=fit, plane_index=-1
            ),
            caustics=include.caustics_from_obj(obj=fit.tracer),
            plotter=sub_plotter,
        )

    elif fit.inversion is not None:

        ag.plot.Inversion.reconstructed_image(
            inversion=fit.inversion,
            light_profile_centres=include.light_profile_centres_from_obj(
                fit.tracer.image_plane
            ),
            mass_profile_centres=include.mass_profile_centres_from_obj(
                fit.tracer.image_plane
            ),
            critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
            image_positions=include.positions_from_fit(fit=fit),
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

        if sub_plotter.figure.aspect in "square":
            aspect_inv = ratio
        elif sub_plotter.figure.aspect in "auto":
            aspect_inv = 1.0 / ratio
        elif sub_plotter.figure.aspect in "equal":
            aspect_inv = 1.0

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=2, aspect=float(aspect_inv)
        )

        ag.plot.Inversion.reconstruction(
            inversion=fit.inversion,
            source_positions=include.positions_of_plane_from_fit_and_plane_index(
                fit=fit, plane_index=-1
            ),
            caustics=include.caustics_from_obj(obj=fit.tracer),
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

    if plot_visibilities:

        ag.plot.FitInterferometer.visibilities(
            fit=fit, include=include, plotter=plotter
        )

    if plot_noise_map:

        ag.plot.FitInterferometer.noise_map(fit=fit, include=include, plotter=plotter)

    if plot_signal_to_noise_map:

        ag.plot.FitInterferometer.signal_to_noise_map(
            fit=fit, include=include, plotter=plotter
        )

    if plot_model_visibilities:

        ag.plot.FitInterferometer.model_visibilities(
            fit=fit, include=include, plotter=plotter
        )

    if plot_residual_map:

        ag.plot.FitInterferometer.residual_map_vs_uv_distances(
            fit=fit, plot_real=True, include=include, plotter=plotter
        )

        ag.plot.FitInterferometer.residual_map_vs_uv_distances(
            fit=fit, plot_real=False, include=include, plotter=plotter
        )

    if plot_normalized_residual_map:

        ag.plot.FitInterferometer.normalized_residual_map_vs_uv_distances(
            fit=fit, plot_real=True, include=include, plotter=plotter
        )

        ag.plot.FitInterferometer.normalized_residual_map_vs_uv_distances(
            fit=fit, plot_real=False, include=include, plotter=plotter
        )

    if plot_chi_squared_map:

        ag.plot.FitInterferometer.chi_squared_map_vs_uv_distances(
            fit=fit, plot_real=True, include=include, plotter=plotter
        )

        ag.plot.FitInterferometer.chi_squared_map_vs_uv_distances(
            fit=fit, plot_real=False, include=include, plotter=plotter
        )
