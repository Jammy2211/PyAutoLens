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
import numpy as np
from typing import Optional, List, Tuple

from autofit.non_linear.grid.sensitivity.result import SensitivityResult

import autofit as af
import autoarray as aa

from autolens.lens.tracer import Tracer


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

