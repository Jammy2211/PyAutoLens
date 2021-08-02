from autofit.non_linear.grid.grid_search import GridSearch as GridSearchAf
from autofit.non_linear.grid.grid_search import GridSearchResult
from autoarray.structures.arrays.two_d.array_2d import Array2D

import numpy as np
from os import path
import json

from typing import List, Tuple

class GridSearch(GridSearchAf):

    def __init__(
            self,
            search,
            result_no_subhalo,
            number_of_steps: int = 4,
            number_of_cores: int = 1,
            result_output_interval: int = 100,
    ):
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
        super().__init__(
            search=search,
            number_of_steps=number_of_steps,
            number_of_cores=number_of_cores,
            result_output_interval=result_output_interval,
            previous_result=result_no_subhalo
        )

    @property
    def result_no_subhalo(self):
        return self.previous_result

    @property
    def grid_search_result_cls(self):
        return SubhaloResult


class SubhaloResult(GridSearchResult):

    @property
    def result_no_subhalo(self):
        return self.previous_result

    def _subhalo_array_from(self, values_native) -> Array2D:

        values_reshaped = [
            value
            for values in values_native
            for value in values
        ]

        return Array2D.manual_yx_and_values(
            y=[centre[0] for centre in self.physical_centres_lists],
            x=[centre[1] for centre in self.physical_centres_lists],
            values=values_reshaped,
            pixel_scales=self.physical_step_sizes,
            shape_native=self.shape_native,
        )

    def subhalo_detection_array_from(
            self,
            use_log_evidences: bool = True,
            use_stochastic_log_evidences: bool = False,
            relative_to_no_subhalo : bool = True,
    ) -> Array2D:

        if (not use_log_evidences) and (not use_stochastic_log_evidences):

            values_native = self.log_likelihoods_native

            if relative_to_no_subhalo:
                values_native -= self.result_no_subhalo.log_likelihood

        elif use_log_evidences and not use_stochastic_log_evidences:

            values_native = self.log_evidences_native

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
    def no_subhalo_stochastic_log_likelihoods(self):

        stochastic_log_evidences_json_file = path.join(
            self.result_no_subhalo.search.paths.output_path, "stochastic_log_evidences.json"
        )

        try:
            with open(stochastic_log_evidences_json_file, "r") as f:
                return np.median(np.asarray(json.load(f)))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found at {self.result_no_subhalo.search.paths.output_path}"
            )

    @property
    def stochastic_log_likelihoods_native(self) -> List[float]:

        stochastic_log_evidences = []

        for result in self.results:

            stochastic_log_evidences_json_file = path.join(
                result.search.paths.output_path, "stochastic_log_evidences.json"
            )

            try:
                with open(stochastic_log_evidences_json_file, "r") as f:
                    stochastic_log_evidences_array = np.asarray(json.load(f))
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"File not found at {result.search.paths.output_path}"
                )

            stochastic_log_evidences.append(np.median(stochastic_log_evidences_array))

        return self._list_to_native(lst=stochastic_log_evidences)

    @property
    def masses_native(self) -> List[float]:
        return self._list_to_native(
            lst=[result.samples.median_pdf_instance.galaxies.subhalo.mass.mass_at_200 for result in self.results])

    @property
    def centres_native(self) -> List[Tuple[float]]:
        return self._list_to_native(
            lst=[result.samples.median_pdf_instance.galaxies.subhalo.mass.centre for result in self.results])