from autolens.aggregator import aggregator as agg

import numpy as np
from autogalaxy.plot import lensing_plotters, plane_plots, inversion_plots
from autolens.plot import fit_imaging_plots
from autoarray.plot import plotters
import os
from os import path
import shutil


def agg_max_log_likelihood_from_aggregator(aggregator):

    samples = list(filter(None, aggregator.values("samples")))
    log_likelihoods = [max(samps.log_likelihoods) for samps in samples]
    index = np.argmax(log_likelihoods)
    search_max = list(filter(None, aggregator.values("search")))[index]
    return aggregator.filter(aggregator.directory.contains(search_max.paths.name))


def copy_pickle_files_to_agg_max(agg_max_log_likelihood):

    search_max_log_likelihood = list(agg_max_log_likelihood.values("search"))
    pickle_path_max_log_likelihood = search_max_log_likelihood[0].paths.pickle_path

    pickle_path_grid_search = path.split(pickle_path_max_log_likelihood)[0]
    pickle_path_grid_search = path.split(pickle_path_grid_search)[0]
    pickle_path_grid_search = path.split(pickle_path_grid_search)[0]
    pickle_path_grid_search = path.split(pickle_path_grid_search)[0]
    pickle_path_grid_search = path.split(pickle_path_grid_search)[0]

    pickle_path_grid_search = path.join(pickle_path_grid_search, "pickles")

    src_files = os.listdir(pickle_path_grid_search)
    for file_name in src_files:
        full_file_name = path.join(pickle_path_grid_search, file_name)
        if path.isfile(full_file_name):
            shutil.copy(full_file_name, pickle_path_max_log_likelihood)


def subplot_detection_agg(
    agg_before, agg_detect, use_log_evidences=True, include=None, sub_plotter=None
):

    fit_imaging_before = list(
        agg.fit_imaging_generator_from_aggregator(aggregator=agg_before)
    )[0]

    agg_max_log_likelihood = agg_max_log_likelihood_from_aggregator(
        aggregator=agg_detect
    )

    copy_pickle_files_to_agg_max(agg_max_log_likelihood=agg_max_log_likelihood)

    fit_imaging_detect = list(
        agg.fit_imaging_generator_from_aggregator(aggregator=agg_max_log_likelihood)
    )[0]

    if use_log_evidences:
        figure_of_merit_before = list(agg_before.values("samples"))[0].log_evidence
    else:
        figure_of_merit_before = fit_imaging_before.figure_of_merit

    detection_array = (
        agg.grid_search_result_as_array(
            aggregator=agg_detect, use_log_evidences=use_log_evidences
        )
        - figure_of_merit_before,
    )[0]

    mass_array = agg.grid_search_subhalo_masses_as_array(aggregator=agg_detect)

    subplot_detection_fits(
        fit_imaging_before=fit_imaging_before,
        fit_imaging_detect=fit_imaging_detect,
        include=include,
        sub_plotter=sub_plotter,
    )

    subplot_detection_imaging(
        fit_imaging_detect=fit_imaging_detect,
        detection_array=detection_array,
        mass_array=mass_array,
        include=include,
        sub_plotter=sub_plotter,
    )


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_detection_fits(
    fit_imaging_before, fit_imaging_detect, include=None, sub_plotter=None
):
    """
    A subplot comparing the normalized residuals, chi-squared map and source reconstructions of the model-fits
    before the subhalo added to the model (top row) and the subhalo fit which gives the largest increase in
    Bayesian evidence on the subhalo detection grid search.

    Parameters
    ----------
    fit_imaging_before : FitImaging
        The fit of a `Tracer` not including a subhalo in the model to a `MaskedImaging` dataset (e.g. the
        model-image, residual_map, chi_squared_map).
    fit_imaging_detect : FitImaging
        The fit of a `Tracer` with the subhalo detection grid's highest evidence model including a subhalo to a
        `MaskedImaging` dataset (e.g. the  model-image, residual_map, chi_squared_map).
    include : Include
        Customizes what appears on the plots (e.g. critical curves, profile centres, origin, etc.).
    sub_plotter : SubPlotter
        Object for plotting PyAutoLens data-stuctures as subplots via Matplotlib.
    """
    number_subplots = 6

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(
        title="Normailzed Residuals (No Subhalo)"
    )

    fit_imaging_plots.normalized_residual_map(
        fit=fit_imaging_before, include=include, plotter=sub_plotter_detect
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(
        title="Chi-Squared Map (No Subhalo)"
    )

    fit_imaging_plots.chi_squared_map(
        fit=fit_imaging_before, include=include, plotter=sub_plotter_detect
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(
        title="Source Reconstruction (No Subhalo)"
    )

    source_model_on_subplot(
        fit=fit_imaging_before,
        plane_index=1,
        number_subplots=6,
        subplot_index=3,
        include=include,
        sub_plotter=sub_plotter_detect,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(
        title="Normailzed Residuals (With Subhalo)"
    )

    fit_imaging_plots.normalized_residual_map(
        fit=fit_imaging_detect, include=include, plotter=sub_plotter_detect
    )

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(
        title="Chi-Squared Map (With Subhalo)"
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    fit_imaging_plots.chi_squared_map(
        fit=fit_imaging_detect, include=include, plotter=sub_plotter_detect
    )

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(
        title="Source Reconstruction (With Subhalo)"
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    source_model_on_subplot(
        fit=fit_imaging_detect,
        plane_index=1,
        number_subplots=6,
        subplot_index=6,
        include=include,
        sub_plotter=sub_plotter_detect,
    )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_detection_imaging(
    fit_imaging_detect, detection_array, mass_array, include=None, sub_plotter=None
):

    number_subplots = 4

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(title="Image")

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    fit_imaging_plots.image(
        fit=fit_imaging_detect, include=include, plotter=sub_plotter_detect
    )

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(
        title="Signal-To-Noise Map"
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    fit_imaging_plots.signal_to_noise_map(
        fit=fit_imaging_detect, include=include, plotter=sub_plotter_detect
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(
        title="Increase in Log Evidence"
    )

    plotters.plot_array(
        array=detection_array, include=include, plotter=sub_plotter_detect
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    sub_plotter_detect = sub_plotter.plotter_with_new_labels(title="Subhalo Mass")

    plotters.plot_array(array=mass_array, include=include, plotter=sub_plotter_detect)

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
