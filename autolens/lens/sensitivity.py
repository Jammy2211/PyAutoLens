import numpy as np
from typing import Optional

import autofit as af
import autoarray as aa
import autogalaxy.plot as aplt

from autoarray.plot.abstract_plotters import AbstractPlotter

from autolens.imaging.fit_imaging import FitImaging
from autolens.imaging.plot.fit_imaging_plotters import FitImagingPlotter


class SubhaloSensitivityResult(af.SensitivityResult):
    def __init__(
        self,
        result_sensitivity: af.SensitivityResult,
    ):
        """
        The results of a subhalo sensitivity mapping analysis, where dark matter halos are used to simulate many
        strong lens datasets which are fitted to quantify how detectable they are.

        Parameters
        ----------
        result_sensitivity
            The results of a sensitivity mapping analysis where.
        """

        super().__init__(
            results=result_sensitivity.results,
            shape=result_sensitivity.shape
        )

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
