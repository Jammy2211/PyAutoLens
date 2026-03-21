"""
Dark-matter subhalo detection via grid searches of non-linear searches.

This module provides result containers and visualisation helpers for a subhalo detection
workflow in which a grid of ``PyAutoFit`` non-linear searches is run.  Each cell of the
grid confines the subhalo's (y, x) centre to a small sub-region of the image plane using
uniform priors and fits the lens model with a subhalo included.

``SubhaloGridSearchResult`` wraps ``af.GridSearchResult`` with:

- ``y`` / ``x`` — the physical centre coordinates of each grid cell.
- ``log_evidence_differences`` — the Bayesian evidence improvement from adding a subhalo
  relative to a smooth-model fit, useful for building a detection significance map.
- Plotting helpers that overlay the detection map on the lens image.
"""
import numpy as np
from typing import List, Optional, Tuple

import autofit as af
import autoarray as aa


class SubhaloGridSearchResult(af.GridSearchResult):
    def __init__(
        self,
        result: af.GridSearchResult,
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
        result
            The results of a grid search of non-linear searches where each DM subhalo's (y,x) coordinates are
            confined to a small region of the image plane via uniform priors.
        """

        super().__init__(
            samples=result.samples,
            lower_limits_lists=result.lower_limits_lists,
            grid_priors=result.grid_priors,
        )

    @property
    def y(self) -> List[float]:
        """
        The y coordinates of the physical values of the subhalo grid, where each value is the centre of a grid cell.

        These are the `centre` coordinates of the dark matter subhalo priors.
        """
        return self.physical_centres_lists_from(
            path="galaxies.subhalo.mass.centre.centre_0"
        )

    @property
    def x(self) -> List[float]:
        """
        The x coordinates of the physical values of the subhalo grid, where each value is the centre of a grid cell.

        These are the `centre` coordinates of the dark matter subhalo priors.
        """
        return self.physical_centres_lists_from(
            path="galaxies.subhalo.mass.centre.centre_1"
        )

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
            y=self.y,
            x=self.x,
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
