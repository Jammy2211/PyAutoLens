import numpy as np
from typing import Optional

import autofit as af
import autoarray as aa
import autogalaxy.plot as aplt

from autoarray.plot.abstract_plotters import AbstractPlotter

from autolens.imaging.fit_imaging import FitImaging
from autolens.imaging.plot.fit_imaging_plotters import FitImagingPlotter


class SubhaloGridSearchResult(af.GridSearchResult):
    def __init__(
        self,
        result_subhalo_grid_search: af.GridSearchResult,
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
        result_subhalo_grid_search
            The results of a grid search of non-linear searches where each DM subhalo's (y,x) coordinates are
            confined to a small region of the image plane via uniform priors.
        """

        super().__init__(
            samples=result_subhalo_grid_search.samples,
            lower_limits_lists=result_subhalo_grid_search.lower_limits_lists,
            grid_priors=result_subhalo_grid_search.grid_priors,
        )

    def _array_2d_from(self, values) -> aa.Array2D:
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
        values_reshaped = [value for values in values.native for value in values]

        return aa.Array2D.from_yx_and_values(
            y=[centre[0] for centre in self.physical_centres_lists],
            x=[centre[1] for centre in self.physical_centres_lists],
            values=values_reshaped,
            pixel_scales=self.physical_step_sizes,
            shape_native=self.shape,
        )

    def figure_of_merit_array(
        self,
        use_log_evidences: bool = True,
        relative_to_value: float = 0.0,
        remove_zeros: bool = False,
    ) -> aa.Array2D:
        """
        Returns an `Array2D` where the values are the figure of merit (`log_evidence` or `log_likelihood`) of every
        lens model on the grid search.

        The values can be computed relative to an input value, `relative_to_value`, which is subtracted from the
        figures of merit. This is typically the figure of merit of the no subhalo model, such that the values
        represent the increase in the figure of merit when a subhalo is included in the lens model and thus
        enable Bayesian model comparison to be performed.

        Values below zero may be rounded to zero, to prevent the figure of merit map being dominated by low values

        Parameters
        ----------
        use_log_evidences
            If `True`, the figure of merit values are the log evidences of every lens model on the grid search.
            If `False`, they are the log likelihoods.
        relative_to_value
            The value to subtract from every figure of merit, which will typically be that of the lens model without
            a so Bayesian model comparison can be easily performed.
        remove_zeros
            If `True`, the figure of merit array is altered so that all values below 0.0 and set to 0.0. For plotting
            relative figures of merit for Bayesian model comparison, this is convenient to remove negative values
            and produce a clearer visualization of the overlay.
        """

        figures_of_merits = self.figure_of_merits(
            use_log_evidences=use_log_evidences, relative_to_value=relative_to_value
        )

        if remove_zeros:
            figures_of_merits = af.GridList(
                values=[fom if fom > 0.0 else 0.0 for fom in figures_of_merits],
                shape=figures_of_merits.shape,
            )

        return self._array_2d_from(values=figures_of_merits)

    @property
    def subhalo_mass_array(self) -> aa.Array2D:
        """
        Returns an `Array2D` where the values are the `mass_at_200` of every DM subhalo of every lens model on the
        grid search.
        """
        return self._array_2d_from(
            values=self.attribute_grid("galaxies.subhalo.mass.mass_at_200")
        )

    @property
    def subhalo_centres_grid(self) -> aa.Grid2D:
        """
        Returns a `Grid2D` where the values are the (y,x) coordinates of every DM subhalo of every lens model on
        the grid search.
        """
        return aa.Grid2D.no_mask(
            values=np.asarray(self.attribute_grid("galaxies.subhalo.mass.centre")),
            pixel_scales=self.physical_step_sizes,
            shape_native=self.shape,
        )


class SubhaloPlotter(AbstractPlotter):
    def __init__(
        self,
        result_subhalo_grid_search: Optional[SubhaloGridSearchResult] = None,
        fit_imaging_with_subhalo: Optional[FitImaging] = None,
        fit_imaging_no_subhalo: Optional[FitImaging] = None,
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
    ):
        """
        Plots the results of scanning for a dark matter subhalo in strong lens imaging.

        This produces the following style of plots:

        - Grid Overlay: The subhalo grid search of non-linear searches fits lens models where the (y,x) coordinates of
          each DM subhalo are confined to a small region of the image plane via uniform priors. Corresponding plots
          overlay the grid of results (e.g. the log evidence increase, subhalo mass) over the images of the fit. This
          provides spatial information of where DM subhalos are detected.

        - Comparison Plots: Plots comparing the results of the model-fit with and without a subhalo, including the
          best-fit lens model, residuals. This illuminates how the inclusion of a subhalo impacts the fit and why the
          DM subhalo is inferred.

        Parameters
        ----------
        result_subhalo_grid_search
            The results of a grid search of non-linear searches where each DM subhalo's (y,x) coordinates are
            confined to a small region of the image plane via uniform priors.
        fit_imaging_with_subhalo
            The `FitImaging` of the model-fit for the lens model with a subhalo (the `subhalo[3]` search in template
            SLaM pipelines).
        fit_imaging_no_subhalo
            The `FitImaging` of the model-fit for the lens model without a subhalo (the `subhalo[1]` search in
            template SLaM pipelines).
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 2D plots.
        """
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.result_subhalo_grid_search = result_subhalo_grid_search

        self.fit_imaging_with_subhalo = fit_imaging_with_subhalo
        self.fit_imaging_no_subhalo = fit_imaging_no_subhalo

    @property
    def fit_imaging_no_subhalo_plotter(self) -> FitImagingPlotter:
        """
        The plotter which plots the results of the model-fit without a subhalo.

        This plot is used in figures such as `subplot_detection_fits` which compare the fits with and without a
        subhalo.
        """
        return FitImagingPlotter(
            fit=self.fit_imaging_no_subhalo,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
        )

    @property
    def fit_imaging_with_subhalo_plotter(self) -> FitImagingPlotter:
        """
        The plotter which plots the results of the model-fit with a subhalo.

        This plot is used in figures such as `subplot_detection_fits` which compare the fits with and without a
        subhalo, or `subplot_detection_imaging` which overlays subhalo grid search results over the image.
        """
        return self.fit_imaging_with_subhalo_plotter_from(visuals_2d=self.visuals_2d)

    def fit_imaging_with_subhalo_plotter_from(self, visuals_2d) -> FitImagingPlotter:
        """
        Returns a plotter of the model-fit with a subhalo, using a specific set of visuals.

        The input visuals are typically the overlay array of the grid search, so that the subhalo grid search results
        can be plotted over the image.

        Parameters
        ----------
        visuals_2d
            The visuals that are plotted over the image of the fit, which are typically the results of the subhalo
            grid search.
        """
        return FitImagingPlotter(
            fit=self.fit_imaging_with_subhalo,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=self.include_2d,
        )

    def set_auto_filename(
        self, filename: str, use_log_evidences: Optional[bool] = None
    ) -> bool:
        """
        If a subplot figure does not have an input filename, this function is used to set one automatically.

        The filename is appended with a string that describes the figure of merit plotted, which is either the
        log evidence or log likelihood.

        Parameters
        ----------
        filename
            The filename of the figure, e.g. 'subhalo_mass'
        use_log_evidences
            If `True`, figures which overlay the goodness-of-fit merit use the `log_evidence`, if `False` the
            `log_likelihood` if used.

        Returns
        -------

        """

        if self.mat_plot_2d.output.filename is None:
            if use_log_evidences is None:
                figure_of_merit = ""
            elif use_log_evidences:
                figure_of_merit = "_log_evidence"
            else:
                figure_of_merit = "_log_likelihood"

            self.set_filename(
                filename=f"{filename}{figure_of_merit}",
            )

            return True

        return False

    def figure_figures_of_merit_grid(
        self,
        use_log_evidences: bool = True,
        relative_to_value: float = 0.0,
        remove_zeros: bool = True,
        show_max_in_title: bool = True,
    ):
        """
        Plot the results of the subhalo grid search, where the figures of merit (e.g. `log_evidence`) of the
        grid search are plotted over the image of the lensed source galaxy.

        The figures of merit can be customized to be relative to the lens model without a subhalo, or with zeros
        rounded up to 0.0 to remove negative values. These produce easily to interpret and visually appealing
        figure of merit overlays.

        Parameters
        ----------
        use_log_evidences
            If `True`, figures which overlay the goodness-of-fit merit use the `log_evidence`, if `False` the
            `log_likelihood` if used.
        relative_to_value
            The value to subtract from every figure of merit, for example which will typically be that of the no
            subhalo lens model so Bayesian model comparison can be easily performed.
        remove_zeros
            If `True`, the figure of merit array is altered so that all values below 0.0 and set to 0.0. For plotting
            relative figures of merit for Bayesian model comparison, this is convenient to remove negative values
            and produce a clearer visualization of the overlay.
        show_max_in_title
            Shows the maximum figure of merit value in the title of the figure, for easy reference.
        """

        reset_filename = self.set_auto_filename(
            filename="subhalo_grid",
            use_log_evidences=use_log_evidences,
        )

        array_overlay = self.result_subhalo_grid_search.figure_of_merit_array(
            use_log_evidences=use_log_evidences,
            relative_to_value=relative_to_value,
            remove_zeros=remove_zeros,
        )

        visuals_2d = self.visuals_2d + self.visuals_2d.__class__(
            array_overlay=array_overlay,
            mass_profile_centres=self.result_subhalo_grid_search.subhalo_centres_grid,
        )

        fit_plotter = self.fit_imaging_with_subhalo_plotter_from(visuals_2d=visuals_2d)

        if show_max_in_title:
            max_value = np.round(np.nanmax(array_overlay), 2)
            fit_plotter.set_title(label=f"Image {max_value}")

        fit_plotter.figures_2d_of_planes(plane_index=-1, subtracted_image=True)

        if reset_filename:
            self.set_filename(filename=None)

    def figure_mass_grid(self):
        """
        Plots the results of the subhalo grid search, where the subhalo mass of every grid search is plotted over
        the image of the lensed source galaxy.
        """

        reset_filename = self.set_auto_filename(
            filename="subhalo_mass",
        )

        array_overlay = self.result_subhalo_grid_search.subhalo_mass_array

        visuals_2d = self.visuals_2d + self.visuals_2d.__class__(
            array_overlay=array_overlay,
            mass_profile_centres=self.result_subhalo_grid_search.subhalo_centres_grid,
        )

        fit_plotter = self.fit_imaging_with_subhalo_plotter_from(visuals_2d=visuals_2d)

        fit_plotter.figures_2d_of_planes(plane_index=-1, subtracted_image=True)

        if reset_filename:
            self.set_filename(filename=None)

    def subplot_detection_imaging(
        self,
        use_log_evidences: bool = True,
        relative_to_value: float = 0.0,
        remove_zeros: bool = False,
    ):
        """
        Plots a subplot showing the image, signal-to-noise-map, figures of merit and subhalo masses of the subhalo
        grid search.

        The figures of merits are plotted as an array, which can be customized to be relative to the lens model without
        a  subhalo, or with zeros rounded up to 0.0 to remove negative values. These produce easily to interpret and
        visually appealing figure of merit overlays.

        Parameters
        ----------
        use_log_evidences
            If `True`, figures which overlay the goodness-of-fit merit use the `log_evidence`, if `False` the
            `log_likelihood` if used.
        relative_to_value
            The value to subtract from every figure of merit, for example which will typically be that of the no
            subhalo lens model so Bayesian model comparison can be easily performed.
        remove_zeros
            If `True`, the figure of merit array is altered so that all values below 0.0 and set to 0.0. For plotting
            relative figures of merit for Bayesian model comparison, this is convenient to remove negative values
            and produce a clearer visualization of the overlay.
        show_max_in_title
            Shows the maximum figure of merit value in the title of the figure, for easy reference.
        """
        self.open_subplot_figure(number_subplots=4)

        self.set_title("Image")
        self.fit_imaging_with_subhalo_plotter.figures_2d(data=True)

        self.set_title("Signal-To-Noise Map")
        self.fit_imaging_with_subhalo_plotter.figures_2d(signal_to_noise_map=True)
        self.set_title(None)

        arr = self.result_subhalo_grid_search.figure_of_merit_array(
            use_log_evidences=use_log_evidences,
            relative_to_value=relative_to_value,
            remove_zeros=remove_zeros,
        )

        self.mat_plot_2d.plot_array(
            array=arr,
            visuals_2d=self.visuals_2d,
            auto_labels=aplt.AutoLabels(title="Increase in Log Evidence"),
        )

        arr = self.result_subhalo_grid_search.subhalo_mass_array

        self.mat_plot_2d.plot_array(
            array=arr,
            visuals_2d=self.visuals_2d,
            auto_labels=aplt.AutoLabels(title="Subhalo Mass"),
        )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_detection_imaging"
        )
        self.close_subplot_figure()

    def subplot_detection_fits(self):
        """
        Plots a subplot comparing the results of the best fit lens models with and without a subhalo.

        This subplot shows the normalized residuals, chi-squared map and source reconstructions of the model-fits
        with and without a subhalo.
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
        self.fit_imaging_with_subhalo_plotter.figures_2d(normalized_residual_map=True)

        self.set_title("Chi-Squared Map (With Subhalo)")
        self.fit_imaging_with_subhalo_plotter.figures_2d(chi_squared_map=True)

        self.set_title("Source Reconstruction (With Subhalo)")
        self.fit_imaging_with_subhalo_plotter.figures_2d_of_planes(
            plane_index=1, plane_image=True
        )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_detection_fits"
        )
        self.close_subplot_figure()
