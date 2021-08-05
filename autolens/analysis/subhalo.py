from autofit.non_linear.grid.grid_search import GridSearch, GridSearchResult
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from typing import Dict, List, Tuple, Optional
import numpy as np


class SubhaloSearch:
    def __init__(self, grid_search: GridSearch, result_no_subhalo):
        """
        Performs a non linear optimiser search for each square in a grid. The dimensionality of the search depends on
        the number of distinct priors passed to the fit function. (1 / step_size) ^ no_dimension steps are performed
        per an optimisation.

        Parameters
        ----------
        number_of_steps
            The number of steps to go in each direction
        search
            The class of the search that is run at each step
        result_output_interval
            The interval between saving a GridSearchResult object via paths
        """
        self.grid_search = grid_search
        self.result_no_subhalo = result_no_subhalo

    def fit(
        self, model, analysis, grid_priors, info: Optional[Dict] = None
    ) -> "SubhaloSearchResult":
        """
        Fit an analysis with a set of grid priors. The grid priors are priors associated with the model mapper
        of this instance that are replaced by uniform priors for each step of the grid search.

        Parameters
        ----------
        model
        analysis: autofit.non_linear.non_linear.Analysis
            An analysis used to determine the fitness of a given model instance
        grid_priors: [p.Prior]
            A list of priors to be substituted for uniform priors across the grid.

        Returns
        -------
        result: GridSearchResult
            An object that comprises the results from each individual fit
        """
        grid_search_result = self.grid_search.fit(
            model=model, analysis=analysis, grid_priors=grid_priors, info=info
        )

        subhalo_search_result = SubhaloSearchResult(
            grid_search_result=grid_search_result,
            result_no_subhalo=self.result_no_subhalo,
        )

        self.grid_search.paths.save_object("result", subhalo_search_result)

        return subhalo_search_result


class SubhaloSearchResult:
    def __init__(self, grid_search_result, result_no_subhalo):

        self.grid_search_result = grid_search_result
        self.result_no_subhalo = result_no_subhalo

    @property
    def model(self):
        return self.grid_search_result.model

    @property
    def instance(self):
        return self.grid_search_result.instance

    def _subhalo_array_from(self, values_native) -> Array2D:

        values_reshaped = [value for values in values_native for value in values]

        return Array2D.manual_yx_and_values(
            y=[centre[0] for centre in self.grid_search_result.physical_centres_lists],
            x=[centre[1] for centre in self.grid_search_result.physical_centres_lists],
            values=values_reshaped,
            pixel_scales=self.grid_search_result.physical_step_sizes,
            shape_native=self.grid_search_result.shape,
        )

    def subhalo_detection_array_from(
        self,
        use_log_evidences: bool = True,
        use_stochastic_log_evidences: bool = False,
        relative_to_no_subhalo: bool = True,
    ) -> Array2D:

        if (not use_log_evidences) and (not use_stochastic_log_evidences):

            values_native = self.grid_search_result.log_likelihoods_native

            if relative_to_no_subhalo:
                values_native -= self.result_no_subhalo.log_likelihood

        elif use_log_evidences and not use_stochastic_log_evidences:

            values_native = self.grid_search_result.log_evidences_native

            if relative_to_no_subhalo:
                values_native -= self.result_no_subhalo.samples.log_evidence

        else:

            values_native = self.stochastic_log_likelihoods_native

            if relative_to_no_subhalo:
                values_native -= self.no_subhalo_stochastic_log_likelihoods

        return self._subhalo_array_from(values_native=values_native)

    def subhalo_mass_array_from(self):
        return self._subhalo_array_from(values_native=self.masses_native)

    @property
    def stochastic_log_likelihoods_native(self) -> List[float]:
        return self.grid_search_result._list_to_native(
            lst=self.stochastic_log_evidences
        )

    @property
    def masses_native(self) -> List[float]:
        return self.grid_search_result._list_to_native(
            lst=[
                result.samples.median_pdf_instance.galaxies.subhalo.mass.mass_at_200
                for result in self.grid_search_result.results
            ]
        )

    @property
    def centres_native(self) -> List[Tuple[float]]:

        centres_native = np.zeros(
            (self.grid_search_result.shape[0], self.grid_search_result.shape[1], 2)
        )

        centres_native[:, :, 0] = self.grid_search_result._list_to_native(
            lst=[
                result.samples.median_pdf_instance.galaxies.subhalo.mass.centre[0]
                for result in self.grid_search_result.results
            ]
        )

        centres_native[:, :, 1] = self.grid_search_result._list_to_native(
            lst=[
                result.samples.median_pdf_instance.galaxies.subhalo.mass.centre[1]
                for result in self.grid_search_result.results
            ]
        )

        return Grid2D.manual_native(
            grid=centres_native,
            pixel_scales=self.grid_search_result.physical_step_sizes,
        )
