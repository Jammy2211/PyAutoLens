from autogalaxy.aggregator import aggregator as ag_agg
from autolens.aggregator import aggregator as agg

import numpy as np
from autoarray.plot import plotters
from autogalaxy.plot import lensing_plotters, plane_plots, inversion_plots
from autolens.plot import fit_imaging_plots
from autoarray.plot import plotters


def agg_max_log_likelihood_from_aggregator(aggregator):

    samples = list(filter(None, aggregator.values("samples")))
    log_likelihoods = [max(samps.log_likelihoods) for samps in samples]
    index = np.argmax(log_likelihoods)
    search_max = list(filter(None, aggregator.values("search")))[index]
    return aggregator.filter(aggregator.directory.contains(search_max.paths.name))


def subplot_detection_agg(agg_detect, agg_before, include=None, sub_plotter=None):

    agg_max_log_likelihood = agg_max_log_likelihood_from_aggregator(
        aggregator=agg_detect
    )
    fit_imaging_detect = list(
        agg.fit_imaging_generator_from_aggregator(aggregator=agg_max_log_likelihood)
    )[0]

    fit_imaging_before = list(
        agg.fit_imaging_generator_from_aggregator(aggregator=agg_before)
    )[0]
    max_log_likelihood_before = fit_imaging_before.figure_of_merit

    detection_array = (
        ag_agg.grid_search_result_as_array(aggregator=agg_detect)
        - max_log_likelihood_before
    )

    subplot_detection_imaging(
        fit_imaging_detect=fit_imaging_detect,
        detection_array=detection_array,
        include=include,
        sub_plotter=sub_plotter,
    )

    subplot_detection_fits(
        fit_imaging_before=fit_imaging_before,
        fit_imaging_detect=fit_imaging_detect,
        include=include,
        sub_plotter=sub_plotter,
    )


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_detection_imaging(
    fit_imaging_detect, detection_array, include=None, sub_plotter=None
):

    number_subplots = 4

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    fit_imaging_plots.image(
        fit=fit_imaging_detect, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    fit_imaging_plots.signal_to_noise_map(
        fit=fit_imaging_detect, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    plotters.plot_array(array=detection_array, include=include, plotter=sub_plotter)

    #  sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    #  fit_imaging_plots.chi_squared_map(fit=fit_imaging_before, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_detection_fits(
    fit_imaging_before, fit_imaging_detect, include=None, sub_plotter=None
):

    number_subplots = 6

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    fit_imaging_plots.normalized_residual_map(
        fit=fit_imaging_before, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    fit_imaging_plots.chi_squared_map(
        fit=fit_imaging_before, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    source_model_on_subplot(
        fit=fit_imaging_before,
        plane_index=1,
        number_subplots=6,
        subplot_index=3,
        include=include,
        sub_plotter=sub_plotter,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    fit_imaging_plots.normalized_residual_map(
        fit=fit_imaging_detect, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    fit_imaging_plots.chi_squared_map(
        fit=fit_imaging_detect, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    source_model_on_subplot(
        fit=fit_imaging_detect,
        plane_index=1,
        number_subplots=6,
        subplot_index=6,
        include=include,
        sub_plotter=sub_plotter,
    )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def source_model_on_subplot(
    fit, plane_index, number_subplots, subplot_index, include, sub_plotter
):

    if not fit.tracer.planes[plane_index].has_pixelization:

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=subplot_index
        )

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
            number_subplots=number_subplots,
            subplot_index=subplot_index,
            aspect=float(aspect_inv),
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
