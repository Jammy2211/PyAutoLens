from typing import List, Tuple
import numpy as np

import autoarray as aa
import autogalaxy.plot as aplt

from autoarray.plot.abstract_plotters import AbstractPlotter

from autolens.imaging.fit_imaging import FitImaging
from autolens.imaging.plot.fit_imaging_plotters import FitImagingPlotter

from autolens.aggregator.fit_imaging import _fit_imaging_from


class SubhaloResult:
    def __init__(
        self, grid_search_result, result_no_subhalo, stochastic_log_likelihoods=None
    ):

        self.grid_search_result = grid_search_result
        self.result_no_subhalo = result_no_subhalo
        self.stochastic_log_likelihoods = stochastic_log_likelihoods

    @property
    def fit_imaging_before(self):
        return _fit_imaging_from(
            fit=self.result_no_subhalo,
            galaxies=self.result_no_subhalo.instance.galaxies,
        )

    def _subhalo_array_from(self, values_native) -> aa.Array2D:

        values_reshaped = [value for values in values_native for value in values]

        return aa.Array2D.manual_yx_and_values(
            y=[centre[0] for centre in self.grid_search_result.physical_centres_lists],
            x=[centre[1] for centre in self.grid_search_result.physical_centres_lists],
            values=values_reshaped,
            pixel_scales=self.grid_search_result.physical_step_sizes,
            shape_native=self.grid_search_result.shape,
        )

    def subhalo_detection_array_from(
        self,
        use_log_evidences: bool = True,
        use_stochastic_log_likelihoods: bool = False,
        relative_to_no_subhalo: bool = True,
    ) -> aa.Array2D:

        if (not use_log_evidences) and (not use_stochastic_log_likelihoods):

            values_native = self.grid_search_result.log_likelihoods_native
            values_native[values_native == None] = np.nan

            if relative_to_no_subhalo:
                values_native -= self.result_no_subhalo[
                    "samples"
                ].max_log_likelihood_sample.log_likelihood

        elif use_log_evidences and not use_stochastic_log_likelihoods:

            values_native = self.grid_search_result.log_evidences_native
            values_native[values_native == None] = np.nan

            if relative_to_no_subhalo:
                values_native -= self.result_no_subhalo["samples"].log_evidence

        else:

            values_native = self.stochastic_log_evidences_native
            values_native[values_native == None] = np.nan

            if relative_to_no_subhalo:
                values_native -= np.median(
                    self.result_no_subhalo["stochastic_log_likelihoods"]
                )

        return self._subhalo_array_from(values_native=values_native)

    def subhalo_mass_array_from(self):
        return self._subhalo_array_from(values_native=self.masses_native)

    @property
    def stochastic_log_evidences_native(self) -> List[float]:

        return self.grid_search_result._list_to_native(
            lst=self.stochastic_log_likelihoods
        )

    def instance_list_via_results_from(self, results):
        return [
            None
            if result.samples.median_pdf_instance is None
            else result.samples.median_pdf_instance
            for result in results
        ]

    @property
    def masses_native(self) -> List[float]:

        instance_list = self.instance_list_via_results_from(
            results=self.grid_search_result.results
        )

        return self.grid_search_result._list_to_native(
            [
                None if instance is None else instance.galaxies.subhalo.mass.mass_at_200
                for instance in instance_list
            ]
        )

    @property
    def centres_native(self) -> List[Tuple[float]]:

        instance_list = self.instance_list_via_results_from(
            results=self.grid_search_result.results
        )

        centres_native = np.zeros(
            (self.grid_search_result.shape[0], self.grid_search_result.shape[1], 2)
        )

        centres_native[:, :, 0] = self.grid_search_result._list_to_native(
            lst=[
                None if instance is None else instance.galaxies.subhalo.mass.centre[0]
                for instance in instance_list
            ]
        )

        centres_native[:, :, 1] = self.grid_search_result._list_to_native(
            lst=[
                None if instance is None else instance.galaxies.subhalo.mass.centre[1]
                for instance in instance_list
            ]
        )

        return aa.Grid2D.manual_native(
            grid=centres_native,
            pixel_scales=self.grid_search_result.physical_step_sizes,
        )


class SubhaloPlotter(AbstractPlotter):
    def __init__(
        self,
        subhalo_result: SubhaloResult,
        fit_imaging_detect,
        use_log_evidences: bool = True,
        use_stochastic_log_likelihoods: bool = False,
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
    ):
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.subhalo_result = subhalo_result
        self.fit_imaging_detect = fit_imaging_detect
        self.use_log_evidences = use_log_evidences
        self.use_stochastic_log_likelihoods = use_stochastic_log_likelihoods

    @property
    def fit_imaging_before(self):
        return self.subhalo_result.fit_imaging_before

    @property
    def fit_imaging_before_plotter(self):
        return FitImagingPlotter(
            fit=self.fit_imaging_before,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
        )

    @property
    def fit_imaging_detect_plotter(self):
        return self.fit_imaging_detect_plotter_from(visuals_2d=self.visuals_2d)

    def fit_imaging_detect_plotter_from(self, visuals_2d):
        return FitImagingPlotter(
            fit=self.fit_imaging_detect,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=self.include_2d,
        )

    def detection_array_from(self, remove_zeros: bool = False):

        detection_array = self.subhalo_result.subhalo_detection_array_from(
            use_log_evidences=self.use_log_evidences,
            use_stochastic_log_likelihoods=self.use_stochastic_log_likelihoods,
            relative_to_no_subhalo=True,
        )

        if remove_zeros:

            detection_array[detection_array < 0.0] = 0.0

        return detection_array

    def figure_with_detection_overlay(
        self,
        image: bool = False,
        remove_zeros: bool = False,
        show_median: bool = True,
        overwrite_title=False,
        transpose_array=False,
    ):

        array_overlay = self.detection_array_from(remove_zeros=remove_zeros)

        median_detection = np.round(np.nanmedian(array_overlay), 2)

        # Due to bug with flipped subhalo inv, can remove one day

        if transpose_array:
            array_overlay = np.fliplr(np.fliplr(array_overlay.native).T)

        visuals_2d = self.visuals_2d + self.visuals_2d.__class__(
            array_overlay=array_overlay,
            mass_profile_centres=self.subhalo_result.centres_native,
        )

        fit_imaging_plotter = self.fit_imaging_detect_plotter_from(
            visuals_2d=visuals_2d
        )

        if show_median:
            if overwrite_title:
                fit_imaging_plotter.set_title(label=f"Image {median_detection}")

        #   fit_imaging_plotter.figures_2d(image=image)
        fit_imaging_plotter.figures_2d_of_planes(plane_index=-1, subtracted_image=True)

    def figure_with_mass_overlay(self, image: bool = False, transpose_array=False):

        array_overlay = self.subhalo_result.subhalo_mass_array_from()

        # Due to bug with flipped subhalo inv, can remove one day

        if transpose_array:
            array_overlay = np.fliplr(np.fliplr(array_overlay.native).T)

        visuals_2d = self.visuals_2d + self.visuals_2d.__class__(
            array_overlay=array_overlay,
            mass_profile_centres=self.subhalo_result.centres_native,
        )

        fit_imaging_plotter = self.fit_imaging_detect_plotter_from(
            visuals_2d=visuals_2d
        )

        fit_imaging_plotter.figures_2d_of_planes(plane_index=-1, subtracted_image=True)

    def subplot_detection_imaging(self, remove_zeros: bool = False):

        self.open_subplot_figure(number_subplots=4)

        self.set_title("Image")
        self.fit_imaging_detect_plotter.figures_2d(image=True)

        self.set_title("Signal-To-Noise Map")
        self.fit_imaging_detect_plotter.figures_2d(signal_to_noise_map=True)
        self.set_title(None)

        self.mat_plot_2d.plot_array(
            array=self.detection_array_from(remove_zeros=remove_zeros),
            visuals_2d=self.visuals_2d,
            auto_labels=aplt.AutoLabels(title="Increase in Log Evidence"),
        )

        mass_array = self.subhalo_result.subhalo_mass_array_from()

        self.mat_plot_2d.plot_array(
            array=mass_array,
            visuals_2d=self.visuals_2d,
            auto_labels=aplt.AutoLabels(title="Subhalo Mass"),
        )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_detection_imaging"
        )
        self.close_subplot_figure()

    def subplot_detection_fits(self):
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
        self.fit_imaging_before_plotter.figures_2d(normalized_residual_map=True)

        self.set_title("Chi-Squared Map (No Subhalo)")
        self.fit_imaging_before_plotter.figures_2d(chi_squared_map=True)

        self.set_title("Source Reconstruction (No Subhalo)")
        self.fit_imaging_before_plotter.figures_2d_of_planes(
            plane_image=True, plane_index=1
        )

        self.set_title("Normailzed Residuals (With Subhalo)")
        self.fit_imaging_detect_plotter.figures_2d(normalized_residual_map=True)

        self.set_title("Chi-Squared Map (With Subhalo)")
        self.fit_imaging_detect_plotter.figures_2d(chi_squared_map=True)

        self.set_title("Source Reconstruction (With Subhalo)")
        self.fit_imaging_detect_plotter.figures_2d_of_planes(
            plane_image=True, plane_index=1
        )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_detection_fits"
        )
        self.close_subplot_figure()
