import numpy as np
from typing import Optional, List, Tuple

from autofit.non_linear.grid.sensitivity.result import SensitivityResult

import autofit as af
import autoarray as aa


from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.auto_labels import AutoLabels

from autolens.lens.tracer import Tracer

import autolens.plot as aplt


class SubhaloSensitivityResult(SensitivityResult):
    def __init__(
        self,
        result: SensitivityResult,
    ):
        """
        The results of a subhalo sensitivity mapping analysis, where dark matter halos are used to simulate many
        strong lens datasets which are fitted to quantify how detectable they are.

        Parameters
        ----------
        result
            The results of a sensitivity mapping analysis where.
        """

        super().__init__(
            samples=result.samples,
            perturb_samples=result.perturb_samples,
            shape=result.shape,
            path_values=result.path_values,
        )

    @property
    def y(self) -> af.GridList:
        """
        The y coordinates of the physical values of the sensitivity mapping grid.

        These are the `centre` coordinates of the dark matter subhalos that are included in the simulated datasets.
        """
        return self.perturbed_physical_centres_list_from(path="mass.centre.centre_0")

    @property
    def x(self) -> af.GridList:
        """
        The x coordinates of the physical values of the sensitivity mapping grid.

        These are the `centre` coordinates of the dark matter subhalos that are included in the simulated datasets.
        """
        return self.perturbed_physical_centres_list_from(path="mass.centre.centre_1")

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """
        The extent of the sensitivity mapping grid, which is the minimum and maximum values of the x and y coordinates.
        """
        return (np.min(self.x), np.max(self.x), np.min(self.y), np.max(self.y))

    def _array_2d_from(self, values) -> aa.Array2D:
        """
        Returns an `Array2D` where the input values are reshaped from list of lists to a 2D array, which is
        suitable for plotting.

        For example, this function may return the 2D array of the increases in log evidence for every lens model
        fitted with a DM subhalo in the sensitivity mapping compared to the model without a DM subhalo.

        The orientation of the 2D array and its values are chosen to ensure that when this array is plotted, DM
        subhalos with positive y and negative x `centre` coordinates appear in the top-left of the image.

        Parameters
        ----------
        values_native
            The list of list of values which are mapped to the 2D array (e.g. the `log_evidence` difference of every
            lens model with a DM subhalo compared to the one without).

        Returns
        -------
        The 2D array of values, where the values are mapped from the input list of lists.
        """
        values_reshaped = [value for values in values.native for value in values]

        pixel_scale_list = []

        for i in range(len(values_reshaped) - 1):
            pixel_scale = abs(self.x[i] - self.x[i + 1])
            if pixel_scale > 0.0:
                pixel_scale_list.append(pixel_scale)

        pixel_scales = np.min(pixel_scale_list)

        return aa.Array2D.from_yx_and_values(
            y=self.y,
            x=self.x,
            values=values_reshaped,
            pixel_scales=(pixel_scales, pixel_scales),
            shape_native=self.shape,
        )

    def figure_of_merit_array(
        self,
        use_log_evidences: bool = True,
        remove_zeros: bool = False,
    ) -> aa.Array2D:
        """
        Returns an `Array2D` where the values are the figure of merit (`log_evidence` or `log_likelihood` difference)
        of every lens model on the sensitivity mapping grid.

        Values below zero may be rounded to zero, to prevent the figure of merit map being dominated by low values

        Parameters
        ----------
        use_log_evidences
            If `True`, the figure of merit values are the log evidences of every lens model on the grid search.
            If `False`, they are the log likelihoods.
        remove_zeros
            If `True`, the figure of merit array is altered so that all values below 0.0 and set to 0.0. For plotting
            relative figures of merit for Bayesian model comparison, this is convenient to remove negative values
            and produce a clearer visualization of the overlay.
        """

        figures_of_merits = self.figure_of_merits(
            use_log_evidences=use_log_evidences,
        )

        if remove_zeros:
            figures_of_merits = af.GridList(
                values=[fom if fom > 0.0 else 0.0 for fom in figures_of_merits],
                shape=figures_of_merits.shape,
            )

        return self._array_2d_from(values=figures_of_merits)


class SubhaloSensitivityPlotter(AbstractPlotter):
    def __init__(
        self,
        mask: Optional[aa.Mask2D] = None,
        tracer_perturb: Optional[Tracer] = None,
        tracer_no_perturb: Optional[Tracer] = None,
        source_image: Optional[aa.Array2D] = None,
        result: Optional[SubhaloSensitivityResult] = None,
        data_subtracted: Optional[aa.Array2D] = None,
        mat_plot_2d: aplt.MatPlot2D = None,
        visuals_2d: aplt.Visuals2D = None,
    ):
        """
        Plots the simulated datasets and results of a sensitivity mapping analysis, where dark matter halos are used
        to simulate many strong lens datasets which are fitted to quantify how detectable they are.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `MassProfile` and plotted via the visuals object.

        Parameters
        ----------
        tracer
            The tracer the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the tracer's light and mass quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """

        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)

        self.mask = mask
        self.tracer_perturb = tracer_perturb
        self.tracer_no_perturb = tracer_no_perturb
        self.source_image = source_image
        self.result = result
        self.data_subtracted = data_subtracted
        self.mat_plot_2d = mat_plot_2d
        self.visuals_2d = visuals_2d

    def update_mat_plot_array_overlay(self, evidence_max):
        evidence_half = evidence_max / 2.0

        self.mat_plot_2d.array_overlay = aplt.ArrayOverlay(
            alpha=0.6, vmin=0.0, vmax=evidence_max
        )
        self.mat_plot_2d.colorbar = aplt.Colorbar(
            manual_tick_values=[0.0, evidence_half, evidence_max],
            manual_tick_labels=[
                0.0,
                np.round(evidence_half, 1),
                np.round(evidence_max, 1),
            ],
        )

    def subplot_tracer_images(self):
        """
        Output the tracer images of the dataset simulated for sensitivity mapping as a .png subplot.

        This dataset corresponds to a single grid-cell on the sensitivity mapping grid and therefore will be output
        many times over the entire sensitivity mapping grid.

        The subplot includes the overall image, the lens galaxy image, the lensed source galaxy image and the source
        galaxy image interpolated to a uniform grid.

        Images are masked before visualization, so that they zoom in on the region of interest which is actually
        fitted.
        """

        grid = aa.Grid2D.from_mask(mask=self.mask)

        image = self.tracer_perturb.image_2d_from(grid=grid)
        lensed_source_image = self.tracer_perturb.image_2d_via_input_plane_image_from(
            grid=grid, plane_image=self.source_image
        )
        lensed_source_image_no_perturb = (
            self.tracer_no_perturb.image_2d_via_input_plane_image_from(
                grid=grid, plane_image=self.source_image
            )
        )

        plotter = aplt.Array2DPlotter(
            array=image,
            mat_plot_2d=self.mat_plot_2d,
        )
        plotter.open_subplot_figure(number_subplots=6)
        plotter.set_title("Image")
        plotter.figure_2d()

        grid = self.mask.derive_grid.unmasked

        visuals_2d = aplt.Visuals2D(
            mask=self.mask,
            tangential_critical_curves=self.tracer_perturb.tangential_critical_curve_list_from(
                grid=grid
            ),
            radial_critical_curves=self.tracer_perturb.radial_critical_curve_list_from(
                grid=grid
            ),
        )

        plotter = aplt.Array2DPlotter(
            array=lensed_source_image,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
        )
        plotter.set_title("Lensed Source Image")
        plotter.figure_2d()

        visuals_2d = aplt.Visuals2D(
            mask=self.mask,
            tangential_caustics=self.tracer_perturb.tangential_caustic_list_from(
                grid=grid
            ),
            radial_caustics=self.tracer_perturb.radial_caustic_list_from(grid=grid),
        )

        plotter = aplt.Array2DPlotter(
            array=self.source_image,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
        )
        plotter.set_title("Source Image")
        plotter.figure_2d()

        plotter = aplt.Array2DPlotter(
            array=self.tracer_perturb.convergence_2d_from(grid=grid),
            mat_plot_2d=self.mat_plot_2d,
        )
        plotter.set_title("Convergence")
        plotter.figure_2d()

        visuals_2d = aplt.Visuals2D(
            mask=self.mask,
            tangential_critical_curves=self.tracer_no_perturb.tangential_critical_curve_list_from(
                grid=grid
            ),
            radial_critical_curves=self.tracer_no_perturb.radial_critical_curve_list_from(
                grid=grid
            ),
        )

        plotter = aplt.Array2DPlotter(
            array=lensed_source_image,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
        )
        plotter.set_title("Lensed Source Image (No Subhalo)")
        plotter.figure_2d()

        residual_map = lensed_source_image - lensed_source_image_no_perturb

        plotter = aplt.Array2DPlotter(
            array=residual_map,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
        )
        plotter.set_title("Residual Map (Subhalo - No Subhalo)")
        plotter.figure_2d()

        plotter.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_lensed_images"
        )
        plotter.close_subplot_figure()

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

    def sensitivity_to_fits(self):
        log_likelihoods = self.result.figure_of_merit_array(
            use_log_evidences=False,
            remove_zeros=False,
        )

        mat_plot_2d = aplt.MatPlot2D(
            output=aplt.Output(
                path=self.mat_plot_2d.output.path,
                filename="sensitivity_log_likelihood",
                format="fits",
            )
        )

        mat_plot_2d.plot_array(
            array=log_likelihoods,
            visuals_2d=self.visuals_2d,
            auto_labels=AutoLabels(),
        )

        try:
            log_evidences = self.result.figure_of_merit_array(
                use_log_evidences=True,
                remove_zeros=False,
            )

            mat_plot_2d = aplt.MatPlot2D(
                output=aplt.Output(
                    path=self.mat_plot_2d.output.path,
                    filename="sensitivity_log_evidence",
                    format="fits",
                )
            )

            mat_plot_2d.plot_array(
                array=log_evidences,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(),
            )

        except TypeError:
            pass

    def subplot_sensitivity(self):
        log_likelihoods = self.result.figure_of_merit_array(
            use_log_evidences=False,
            remove_zeros=True,
        )

        try:
            log_evidences = self.result.figure_of_merit_array(
                use_log_evidences=True,
                remove_zeros=True,
            )
        except TypeError:
            log_evidences = np.zeros_like(log_likelihoods)

        self.open_subplot_figure(number_subplots=8, subplot_shape=(2, 4))

        plotter = aplt.Array2DPlotter(
            array=self.data_subtracted,
            mat_plot_2d=self.mat_plot_2d,
        )

        plotter.figure_2d()

        self.mat_plot_2d.plot_array(
            array=log_evidences,
            visuals_2d=self.visuals_2d,
            auto_labels=AutoLabels(title="Increase in Log Evidence"),
        )

        self.mat_plot_2d.plot_array(
            array=log_likelihoods,
            visuals_2d=self.visuals_2d,
            auto_labels=AutoLabels(title="Increase in Log Likelihood"),
        )

        above_threshold = np.where(log_likelihoods > 5.0, 1.0, 0.0)

        above_threshold = aa.Array2D(values=above_threshold, mask=log_likelihoods.mask)

        self.mat_plot_2d.plot_array(
            array=above_threshold,
            visuals_2d=self.visuals_2d,
            auto_labels=AutoLabels(title="Log Likelihood > 5.0"),
        )

        try:
            log_evidences_base = self.result._array_2d_from(
                self.result.log_evidences_base
            )
            log_evidences_perturbed = self.result._array_2d_from(
                self.result.log_evidences_perturbed
            )

            log_evidences_base_min = np.nanmin(
                np.where(log_evidences_base == 0, np.nan, log_evidences_base)
            )
            log_evidences_base_max = np.nanmax(
                np.where(log_evidences_base == 0, np.nan, log_evidences_base)
            )
            log_evidences_perturbed_min = np.nanmin(
                np.where(log_evidences_perturbed == 0, np.nan, log_evidences_perturbed)
            )
            log_evidences_perturbed_max = np.nanmax(
                np.where(log_evidences_perturbed == 0, np.nan, log_evidences_perturbed)
            )

            self.mat_plot_2d.cmap.kwargs["vmin"] = np.min(
                [log_evidences_base_min, log_evidences_perturbed_min]
            )
            self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(
                [log_evidences_base_max, log_evidences_perturbed_max]
            )

            self.mat_plot_2d.plot_array(
                array=log_evidences_base,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(title="Log Evidence Base"),
            )

            self.mat_plot_2d.plot_array(
                array=log_evidences_perturbed,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(title="Log Evidence Perturb"),
            )
        except TypeError:
            pass

        log_likelihoods_base = self.result._array_2d_from(
            self.result.log_likelihoods_base
        )
        log_likelihoods_perturbed = self.result._array_2d_from(
            self.result.log_likelihoods_perturbed
        )

        log_likelihoods_base_min = np.nanmin(
            np.where(log_likelihoods_base == 0, np.nan, log_likelihoods_base)
        )
        log_likelihoods_base_max = np.nanmax(
            np.where(log_likelihoods_base == 0, np.nan, log_likelihoods_base)
        )
        log_likelihoods_perturbed_min = np.nanmin(
            np.where(log_likelihoods_perturbed == 0, np.nan, log_likelihoods_perturbed)
        )
        log_likelihoods_perturbed_max = np.nanmax(
            np.where(log_likelihoods_perturbed == 0, np.nan, log_likelihoods_perturbed)
        )

        self.mat_plot_2d.cmap.kwargs["vmin"] = np.min(
            [log_likelihoods_base_min, log_likelihoods_perturbed_min]
        )
        self.mat_plot_2d.cmap.kwargs["vmax"] = np.max(
            [log_likelihoods_base_max, log_likelihoods_perturbed_max]
        )

        self.mat_plot_2d.plot_array(
            array=log_likelihoods_base,
            visuals_2d=self.visuals_2d,
            auto_labels=AutoLabels(title="Log Likelihood Base"),
        )

        self.mat_plot_2d.plot_array(
            array=log_likelihoods_perturbed,
            visuals_2d=self.visuals_2d,
            auto_labels=AutoLabels(title="Log Likelihood Perturb"),
        )

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_sensitivity")

        self.close_subplot_figure()

    def subplot_figures_of_merit_grid(
        self,
        use_log_evidences: bool = True,
        remove_zeros: bool = True,
        show_max_in_title: bool = True,
    ):
        self.open_subplot_figure(number_subplots=1)

        figures_of_merit = self.result.figure_of_merit_array(
            use_log_evidences=use_log_evidences,
            remove_zeros=remove_zeros,
        )

        if show_max_in_title:
            max_value = np.round(np.nanmax(figures_of_merit), 2)
            self.set_title(label=f"Sensitivity Map {max_value}")

        self.update_mat_plot_array_overlay(evidence_max=np.max(figures_of_merit))

        self.mat_plot_2d.plot_array(
            array=figures_of_merit,
            visuals_2d=self.visuals_2d,
            auto_labels=AutoLabels(title="Increase in Log Evidence"),
        )

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="sensitivity")
        self.close_subplot_figure()

    def figure_figures_of_merit_grid(
        self,
        use_log_evidences: bool = True,
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
            filename="sensitivity",
            use_log_evidences=use_log_evidences,
        )

        array_overlay = self.result.figure_of_merit_array(
            use_log_evidences=use_log_evidences,
            remove_zeros=remove_zeros,
        )

        visuals_2d = self.visuals_2d + self.visuals_2d.__class__(
            array_overlay=array_overlay,
        )

        self.update_mat_plot_array_overlay(evidence_max=np.max(array_overlay))

        plotter = aplt.Array2DPlotter(
            array=self.data_subtracted,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
        )

        if show_max_in_title:
            max_value = np.round(np.nanmax(array_overlay), 2)
            plotter.set_title(label=f"Sensitivity Map {max_value}")

        plotter.figure_2d()

        if reset_filename:
            self.set_filename(filename=None)
