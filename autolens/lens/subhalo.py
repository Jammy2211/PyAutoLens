from typing import List
import numpy as np

import autofit as af
import autoarray as aa
import autogalaxy.plot as aplt

from autoarray.plot.abstract_plotters import AbstractPlotter

from autolens.imaging.fit_imaging import FitImaging
from autolens.imaging.plot.fit_imaging_plotters import FitImagingPlotter


class SubhaloResult:
    def __init__(
        self,
        grid_search_result_with_subhalo : af.GridSearchResult,
        fit_imaging_no_subhalo : FitImaging,
        samples_no_subhalo : af.Samples,
    ):
        """
        The results of a subhalo detection analysis, where dark matter halos are added to the lens model and fitted
        to the data.

        This result may use a grid search of non-linear searches where the (y,x) coordinates of each DM subhalo
        included in the lens model are confined to a small region of the image plane via uniform priors. This object
        contains functionality for creates ndarrays of these results for visualization and analysis.

        The samples of a previous lens model fit, not including a subhalo, may also be passed to this object. These
        are used to plot all quantities relative to the no subhalo model, e.g. the change in log evidence.

        Parameters
        ----------
        grid_search_result_with_subhalo
            The results of a grid search of non-linear searches where each DM subhalo's (y,x) coordinates are
            confined to a small region of the image plane via uniform priors.
        fit_imaging_no_subhalo
            The `FitImaging` of the model-fit to the image without a subhalo.
        samples_no_subhalo
            The `Samples` of the model-fit to the image without a subhalo.
        """
        self.grid_search_result_with_subhalo = grid_search_result_with_subhalo
        self.fit_imaging_no_subhalo = fit_imaging_no_subhalo
        self.samples_no_subhalo = samples_no_subhalo

    def _array_2d_from(self, values_native) -> aa.Array2D:
        """
        Returns an `Array2D` where the input values are reshaped from list of lists to a 2D array, which is
        suitable for plotting.

        For example, this function may return the 2D array of the increases in log evidence for every lens model
        with a DM subhalo.

        The orientation of the 2D array and its values are chosen to ensure that when this array is plotted, DM
        subhalos with positive y and negative x `centre` coordinates appear in the top-left of the image.

        Parameters
        ----------
        values_native
            The list of list of values which are mapped to the 2D array (e.g. the `log_evidence` increases of every
            lens model with a DM subhalo).

        Returns
        -------
        The 2D array of values, where the values are mapped from the input list of lists.

        """
        values_reshaped = [value for values in values_native for value in values]

        return aa.Array2D.from_yx_and_values(
            y=[centre[0] for centre in self.grid_search_result_with_subhalo.physical_centres_lists],
            x=[centre[1] for centre in self.grid_search_result_with_subhalo.physical_centres_lists],
            values=values_reshaped,
            pixel_scales=self.grid_search_result_with_subhalo.physical_step_sizes,
            shape_native=self.grid_search_result_with_subhalo.shape,
        )

    def subhalo_detection_array_from(
        self,
        use_log_evidences: bool = True,
        relative_to_no_subhalo: bool = True,
    ) -> aa.Array2D:

        if not use_log_evidences:
            values_native = self.grid_search_result_with_subhalo.log_likelihoods_native
            values_native[values_native == None] = np.nan

            if relative_to_no_subhalo:
                values_native -= (
                    self.samples_no_subhalo.max_log_likelihood_sample.log_likelihood
                )

        elif use_log_evidences:
            values_native = self.grid_search_result_with_subhalo.log_evidences_native
            values_native[values_native == None] = np.nan

            if relative_to_no_subhalo:
                values_native -= self.samples_no_subhalo.log_evidence

        return self._array_2d_from(values_native=values_native)

    def subhalo_mass_array_from(self):
        return self._array_2d_from(values_native=self.masses_native)

    def instance_list_via_results_from(self, results):
        return [
            None if result.samples.median_pdf() is None else result.samples.median_pdf()
            for result in results
        ]

    @property
    def masses_native(self) -> List[float]:
        instance_list = self.instance_list_via_results_from(
            results=self.grid_search_result_with_subhalo.results
        )

        return self.grid_search_result_with_subhalo._list_to_native(
            [
                None if instance is None else instance.galaxies.subhalo.mass.mass_at_200
                for instance in instance_list
            ]
        )

    @property
    def centres_native(self) -> aa.Grid2D:
        instance_list = self.instance_list_via_results_from(
            results=self.grid_search_result_with_subhalo.results
        )

        centres_native = np.zeros(
            (self.grid_search_result_with_subhalo.shape[0], self.grid_search_result_with_subhalo.shape[1], 2)
        )

        centres_native[:, :, 0] = self.grid_search_result_with_subhalo._list_to_native(
            lst=[
                None if instance is None else instance.galaxies.subhalo.mass.centre[0]
                for instance in instance_list
            ]
        )

        centres_native[:, :, 1] = self.grid_search_result_with_subhalo._list_to_native(
            lst=[
                None if instance is None else instance.galaxies.subhalo.mass.centre[1]
                for instance in instance_list
            ]
        )

        return aa.Grid2D.no_mask(
            values=centres_native,
            pixel_scales=self.grid_search_result_with_subhalo.physical_step_sizes,
        )


class SubhaloPlotter(AbstractPlotter):
    def __init__(
        self,
        subhalo_result: SubhaloResult,
        fit_imaging_detect,
        use_log_evidences: bool = True,
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

    @property
    def fit_imaging_no_subhalo_plotter(self):
        return FitImagingPlotter(
            fit=self.subhalo_result.fit_imaging_no_subhalo,
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

        fit_plotter = self.fit_imaging_detect_plotter_from(visuals_2d=visuals_2d)

        if show_median:
            if overwrite_title:
                fit_plotter.set_title(label=f"Image {median_detection}")

        #   fit_plotter.figures_2d(image=image)
        fit_plotter.figures_2d_of_planes(plane_index=-1, subtracted_image=True)

    def figure_with_mass_overlay(self, image: bool = False, transpose_array=False):
        array_overlay = self.subhalo_result.subhalo_mass_array_from()

        # Due to bug with flipped subhalo inv, can remove one day

        if transpose_array:
            array_overlay = np.fliplr(np.fliplr(array_overlay.native).T)

        visuals_2d = self.visuals_2d + self.visuals_2d.__class__(
            array_overlay=array_overlay,
            mass_profile_centres=self.subhalo_result.centres_native,
        )

        fit_plotter = self.fit_imaging_detect_plotter_from(visuals_2d=visuals_2d)

        fit_plotter.figures_2d_of_planes(plane_index=-1, subtracted_image=True)

    def subplot_detection_imaging(self, remove_zeros: bool = False):
        self.open_subplot_figure(number_subplots=4)

        self.set_title("Image")
        self.fit_imaging_detect_plotter.figures_2d(data=True)

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
        self.fit_imaging_no_subhalo_plotter.figures_2d(normalized_residual_map=True)

        self.set_title("Chi-Squared Map (No Subhalo)")
        self.fit_imaging_no_subhalo_plotter.figures_2d(chi_squared_map=True)

        self.set_title("Source Reconstruction (No Subhalo)")
        self.fit_imaging_no_subhalo_plotter.figures_2d_of_planes(
            plane_index=1, plane_image=True
        )

        self.set_title("Normailzed Residuals (With Subhalo)")
        self.fit_imaging_detect_plotter.figures_2d(normalized_residual_map=True)

        self.set_title("Chi-Squared Map (With Subhalo)")
        self.fit_imaging_detect_plotter.figures_2d(chi_squared_map=True)

        self.set_title("Source Reconstruction (With Subhalo)")
        self.fit_imaging_detect_plotter.figures_2d_of_planes(
            plane_index=1, plane_image=True
        )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_detection_fits"
        )
        self.close_subplot_figure()
