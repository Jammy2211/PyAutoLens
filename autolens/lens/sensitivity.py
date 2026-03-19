"""
Sensitivity mapping for dark-matter subhalo detection.

Sensitivity mapping asks: *given a lens model, how detectable would a dark-matter subhalo
of mass M at position (y, x) be?*  It works by:

1. Fixing the smooth lens model to the best-fit result.
2. Simulating many lens datasets, each with a subhalo at a different position and mass,
   using the smooth model plus a perturbation.
3. Fitting each simulated dataset with the smooth model (no subhalo) to measure the change
   in log-evidence caused by the subhalo.

``SubhaloSensitivityResult`` wraps the generic ``PyAutoFit`` ``SensitivityResult`` with
convenience properties for the subhalo grid positions (``y``, ``x``), the detection
significance map, and Matplotlib visualisation helpers.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple

from autofit.non_linear.grid.sensitivity.result import SensitivityResult

import autofit as af
import autoarray as aa

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autogalaxy.plot.abstract_plotters import _save_subplot

from autolens.plot.abstract_plotters import Plotter as AbstractPlotter

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
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.mask = mask
        self.tracer_perturb = tracer_perturb
        self.tracer_no_perturb = tracer_no_perturb
        self.source_image = source_image
        self.result = result
        self.data_subtracted = data_subtracted

    def subplot_tracer_images(self):
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

        unmasked_grid = self.mask.derive_grid.unmasked

        from autolens.lens.tracer_util import critical_curves_from, caustics_from

        tan_cc_p, rad_cc_p = critical_curves_from(tracer=self.tracer_perturb, grid=unmasked_grid)
        perturb_cc_lines = [
            np.array(c.array if hasattr(c, "array") else c)
            for c in list(tan_cc_p) + list(rad_cc_p)
        ] or None

        tan_ca_p, rad_ca_p = caustics_from(tracer=self.tracer_perturb, grid=unmasked_grid)
        perturb_ca_lines = [
            np.array(c.array if hasattr(c, "array") else c)
            for c in list(tan_ca_p) + list(rad_ca_p)
        ] or None

        tan_cc_n, rad_cc_n = critical_curves_from(tracer=self.tracer_no_perturb, grid=unmasked_grid)
        no_perturb_cc_lines = [
            np.array(c.array if hasattr(c, "array") else c)
            for c in list(tan_cc_n) + list(rad_cc_n)
        ] or None

        residual_map = lensed_source_image - lensed_source_image_no_perturb

        fig, axes = plt.subplots(1, 6, figsize=(42, 7))

        aplt.Array2DPlotter(array=image, output=self.output, cmap=self.cmap, use_log10=self.use_log10).figure_2d(ax=axes[0])
        axes[0].set_title("Image")

        aplt.Array2DPlotter(array=lensed_source_image, output=self.output, cmap=self.cmap, use_log10=self.use_log10, lines=perturb_cc_lines).figure_2d(ax=axes[1])
        axes[1].set_title("Lensed Source Image")

        aplt.Array2DPlotter(array=self.source_image, output=self.output, cmap=self.cmap, use_log10=self.use_log10, lines=perturb_ca_lines).figure_2d(ax=axes[2])
        axes[2].set_title("Source Image")

        aplt.Array2DPlotter(array=self.tracer_perturb.convergence_2d_from(grid=grid), output=self.output, cmap=self.cmap, use_log10=self.use_log10).figure_2d(ax=axes[3])
        axes[3].set_title("Convergence")

        aplt.Array2DPlotter(array=lensed_source_image, output=self.output, cmap=self.cmap, use_log10=self.use_log10, lines=no_perturb_cc_lines).figure_2d(ax=axes[4])
        axes[4].set_title("Lensed Source Image (No Subhalo)")

        aplt.Array2DPlotter(array=residual_map, output=self.output, cmap=self.cmap, use_log10=self.use_log10, lines=no_perturb_cc_lines).figure_2d(ax=axes[5])
        axes[5].set_title("Residual Map (Subhalo - No Subhalo)")

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_lensed_images")

    def set_auto_filename(
        self, filename: str, use_log_evidences: Optional[bool] = None
    ) -> bool:
        if self.output.filename is None:
            if use_log_evidences is None:
                figure_of_merit = ""
            elif use_log_evidences:
                figure_of_merit = "_log_evidence"
            else:
                figure_of_merit = "_log_likelihood"

            self.set_filename(filename=f"{filename}{figure_of_merit}")
            return True

        return False

    def sensitivity_to_fits(self):
        log_likelihoods = self.result.figure_of_merit_array(
            use_log_evidences=False,
            remove_zeros=False,
        )

        fits_output = Output(
            path=self.output.path,
            filename="sensitivity_log_likelihood",
            format="fits",
        )
        aplt.Array2DPlotter(array=log_likelihoods, output=fits_output).figure_2d()

        try:
            log_evidences = self.result.figure_of_merit_array(
                use_log_evidences=True,
                remove_zeros=False,
            )

            fits_output = Output(
                path=self.output.path,
                filename="sensitivity_log_evidence",
                format="fits",
            )
            aplt.Array2DPlotter(array=log_evidences, output=fits_output).figure_2d()

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

        above_threshold = np.where(log_likelihoods > 5.0, 1.0, 0.0)
        above_threshold = aa.Array2D(values=above_threshold, mask=log_likelihoods.mask)

        fig, axes = plt.subplots(2, 4, figsize=(28, 14))
        axes_flat = list(axes.flatten())

        aplt.Array2DPlotter(array=self.data_subtracted, output=self.output, cmap=self.cmap, use_log10=self.use_log10).figure_2d(ax=axes_flat[0])

        self._plot_array(array=log_evidences, auto_filename="increase_in_log_evidence", title="Increase in Log Evidence", ax=axes_flat[1])
        self._plot_array(array=log_likelihoods, auto_filename="increase_in_log_likelihood", title="Increase in Log Likelihood", ax=axes_flat[2])
        self._plot_array(array=above_threshold, auto_filename="log_likelihood_above_5", title="Log Likelihood > 5.0", ax=axes_flat[3])

        ax_idx = 4
        try:
            log_evidences_base = self.result._array_2d_from(self.result.log_evidences_base)
            log_evidences_perturbed = self.result._array_2d_from(self.result.log_evidences_perturbed)

            log_evidences_base_min = np.nanmin(np.where(log_evidences_base == 0, np.nan, log_evidences_base))
            log_evidences_base_max = np.nanmax(np.where(log_evidences_base == 0, np.nan, log_evidences_base))
            log_evidences_perturbed_min = np.nanmin(np.where(log_evidences_perturbed == 0, np.nan, log_evidences_perturbed))
            log_evidences_perturbed_max = np.nanmax(np.where(log_evidences_perturbed == 0, np.nan, log_evidences_perturbed))

            self.cmap.kwargs["vmin"] = np.min([log_evidences_base_min, log_evidences_perturbed_min])
            self.cmap.kwargs["vmax"] = np.max([log_evidences_base_max, log_evidences_perturbed_max])

            self._plot_array(array=log_evidences_base, auto_filename="log_evidence_base", title="Log Evidence Base", ax=axes_flat[ax_idx])
            ax_idx += 1
            self._plot_array(array=log_evidences_perturbed, auto_filename="log_evidence_perturb", title="Log Evidence Perturb", ax=axes_flat[ax_idx])
            ax_idx += 1
        except TypeError:
            pass

        log_likelihoods_base = self.result._array_2d_from(self.result.log_likelihoods_base)
        log_likelihoods_perturbed = self.result._array_2d_from(self.result.log_likelihoods_perturbed)

        log_likelihoods_base_min = np.nanmin(np.where(log_likelihoods_base == 0, np.nan, log_likelihoods_base))
        log_likelihoods_base_max = np.nanmax(np.where(log_likelihoods_base == 0, np.nan, log_likelihoods_base))
        log_likelihoods_perturbed_min = np.nanmin(np.where(log_likelihoods_perturbed == 0, np.nan, log_likelihoods_perturbed))
        log_likelihoods_perturbed_max = np.nanmax(np.where(log_likelihoods_perturbed == 0, np.nan, log_likelihoods_perturbed))

        self.cmap.kwargs["vmin"] = np.min([log_likelihoods_base_min, log_likelihoods_perturbed_min])
        self.cmap.kwargs["vmax"] = np.max([log_likelihoods_base_max, log_likelihoods_perturbed_max])

        self._plot_array(array=log_likelihoods_base, auto_filename="log_likelihood_base", title="Log Likelihood Base", ax=axes_flat[ax_idx])
        ax_idx += 1
        self._plot_array(array=log_likelihoods_perturbed, auto_filename="log_likelihood_perturb", title="Log Likelihood Perturb", ax=axes_flat[ax_idx])

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_sensitivity")

    def subplot_figures_of_merit_grid(
        self,
        use_log_evidences: bool = True,
        remove_zeros: bool = True,
        show_max_in_title: bool = True,
    ):
        figures_of_merit = self.result.figure_of_merit_array(
            use_log_evidences=use_log_evidences,
            remove_zeros=remove_zeros,
        )

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        if show_max_in_title:
            max_value = np.round(np.nanmax(figures_of_merit), 2)
            ax.set_title(f"Sensitivity Map {max_value}")

        self._plot_array(array=figures_of_merit, auto_filename="sensitivity", title="Increase in Log Evidence", ax=ax)

        plt.tight_layout()
        _save_subplot(fig, self.output, "sensitivity")

    def figure_figures_of_merit_grid(
        self,
        use_log_evidences: bool = True,
        remove_zeros: bool = True,
        show_max_in_title: bool = True,
    ):
        reset_filename = self.set_auto_filename(
            filename="sensitivity",
            use_log_evidences=use_log_evidences,
        )

        array_overlay = self.result.figure_of_merit_array(
            use_log_evidences=use_log_evidences,
            remove_zeros=remove_zeros,
        )

        plotter = aplt.Array2DPlotter(
            array=self.data_subtracted,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            array_overlay=array_overlay,
        )

        if show_max_in_title:
            max_value = np.round(np.nanmax(array_overlay), 2)
            plotter.set_title(label=f"Sensitivity Map {max_value}")

        plotter.figure_2d()

        if reset_filename:
            self.set_filename(filename=None)
