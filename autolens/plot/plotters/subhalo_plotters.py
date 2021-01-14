from autoarray.plot.mat_wrap import mat_plot as mp
from autolens.aggregator import aggregator as agg
from autoarray.plot.plotters import abstract_plotters
import numpy as np
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autolens.plot.plotters import fit_imaging_plotters
import os
from os import path
import shutil


class SubhaloPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

    def fit_imaging_plotter_from(self, fit_imaging):
        return fit_imaging_plotters.FitImagingPlotter(
            fit=fit_imaging,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
        )

    def subplot_detection_imaging(
        self, fit_imaging_detect, detection_array, mass_array
    ):

        self.open_subplot_figure(number_subplots=4)

        self.set_title("Image")
        fit_imaging_plotter = self.fit_imaging_plotter_from(
            fit_imaging=fit_imaging_detect
        )
        fit_imaging_plotter.figures(image=True)

        self.set_title("Signal-To-Noise Map")
        fit_imaging_plotter.figures(signal_to_noise_map=True)

        self.mat_plot_2d.plot_array(
            array=detection_array,
            visuals_2d=self.visuals_2d,
            auto_labels=mp.AutoLabels(title="Increase in Log Evidence"),
        )

        self.mat_plot_2d.plot_array(
            array=mass_array,
            visuals_2d=self.visuals_2d,
            auto_labels=mp.AutoLabels(title="Subhalo Mass"),
        )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_detection_imaging"
        )
        self.close_subplot_figure()

    def subplot_detection_fits(self, fit_imaging_before, fit_imaging_detect):
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
        mat_plot_2d : Plotter
            Object for plotting PyAutoLens data-stuctures as subplots via Matplotlib.
        """

        self.open_subplot_figure(number_subplots=6)

        self.set_title("Normalized Residuals (No Subhalo)")
        fit_imaging_plotter_before = self.fit_imaging_plotter_from(
            fit_imaging=fit_imaging_before
        )
        fit_imaging_plotter_before.figures(normalized_residual_map=True)

        self.set_title("Chi-Squared Map (No Subhalo)")
        fit_imaging_plotter_before.figures(chi_squared_map=True)

        self.set_title("Source Reconstruction (No Subhalo)")
        fit_imaging_plotter_before.figures_of_planes(plane_image=True, plane_index=1)
        fit_imaging_plotter_detect = self.fit_imaging_plotter_from(
            fit_imaging=fit_imaging_detect
        )

        self.set_title("Normailzed Residuals (With Subhalo)")
        fit_imaging_plotter_detect.figures(normalized_residual_map=True)

        self.set_title("Chi-Squared Map (With Subhalo)")
        fit_imaging_plotter_detect.figures(chi_squared_map=True)

        self.set_title("Source Reconstruction (With Subhalo)")
        fit_imaging_plotter_detect.figures_of_planes(plane_image=True, plane_index=1)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_detection_fits"
        )
        self.close_subplot_figure()


def agg_max_log_likelihood_from_aggregator(aggregator):

    samples = list(filter(None, aggregator.values("samples")))
    log_likelihoods = [max(samps.log_likelihoods) for samps in samples]
    index = np.argmax(log_likelihoods)
    search_max = list(filter(None, aggregator.values("search")))[index]

    directory = str(search_max.paths.name)
    directory = directory.replace(r"/", path.sep)

    return aggregator.filter(aggregator.directory.contains(directory))


def copy_pickle_files_to_agg_max(agg_max_log_likelihood):

    search_max_log_likelihood = list(agg_max_log_likelihood.values("search"))
    pickle_path_max_log_likelihood = search_max_log_likelihood[0].paths.pickle_path

    pickle_path_max_log_likelihood = str(pickle_path_max_log_likelihood).replace(
        r"/", path.sep
    )

    pickle_path_grid_search = pickle_path_max_log_likelihood

    pickle_path_grid_search = path.split(pickle_path_grid_search)[0]
    pickle_path_grid_search = path.split(pickle_path_grid_search)[0]
    pickle_path_grid_search = path.split(pickle_path_grid_search)[0]

    # TODO : needed for linux?

    #   pickle_path_grid_search = path.split(pickle_path_grid_search)[0]
    #   pickle_path_grid_search = path.split(pickle_path_grid_search)[0]

    pickle_path_grid_search = path.join(pickle_path_grid_search, "pickles")

    src_files = os.listdir(pickle_path_grid_search)

    for file_name in src_files:
        full_file_name = path.join(pickle_path_grid_search, file_name)
        if path.isfile(full_file_name):
            shutil.copy(full_file_name, pickle_path_max_log_likelihood)


def detection_array_from(
    agg_before, agg_detect, use_log_evidences=True, use_stochastic_log_evidences=False
):

    fit_imaging_before = list(
        agg.fit_imaging_generator_from_aggregator(aggregator=agg_before)
    )[0]

    if use_log_evidences and not use_stochastic_log_evidences:
        figure_of_merit_before = list(agg_before.values("samples"))[0].log_evidence
    elif use_stochastic_log_evidences:
        figure_of_merit_before = np.median(
            list(agg_before.values("stochastic_log_evidences"))[0]
        )
    else:
        figure_of_merit_before = fit_imaging_before.figure_of_merit

    return (
        agg.grid_search_result_as_array(
            aggregator=agg_detect,
            use_log_evidences=use_log_evidences,
            use_stochastic_log_evidences=use_stochastic_log_evidences,
        )
        - figure_of_merit_before,
    )[0]


def mass_array_from(agg_detect):
    return agg.grid_search_subhalo_masses_as_array(aggregator=agg_detect)


def subplot_detection_agg(
    agg_before,
    agg_detect,
    use_log_evidences=True,
    use_stochastic_log_evidences=False,
    subhalo_plotter=SubhaloPlotter(),
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

    detection_array = detection_array_from(
        agg_before=agg_before,
        agg_detect=agg_detect,
        use_log_evidences=use_log_evidences,
        use_stochastic_log_evidences=use_stochastic_log_evidences,
    )

    mass_array = mass_array_from(agg_detect=agg_detect)

    subhalo_plotter.subplot_detection_fits(
        fit_imaging_before=fit_imaging_before, fit_imaging_detect=fit_imaging_detect
    )

    subhalo_plotter.subplot_detection_imaging(
        fit_imaging_detect=fit_imaging_detect,
        detection_array=detection_array,
        mass_array=mass_array,
    )
