from autofit.non_linear.grid.grid_search import GridSearch as GridSearchAf
from autofit.non_linear.grid.grid_search import GridSearchResult
from autoarray.structures.arrays.two_d.array_2d import Array2D

import numpy as np
from os import path
import json

from typing import List, Tuple

class GridSearch(GridSearchAf):

    @property
    def grid_search_result_cls(self):
        return SubhaloResult


class SubhaloResult(GridSearchResult):
    
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
    ) -> Array2D:
        if (not use_log_evidences) and (not use_stochastic_log_evidences):
            return self._subhalo_array_from(values_native=self.log_likelihoods_native)
        elif use_log_evidences and not use_stochastic_log_evidences:
            return self._subhalo_array_from(values_native=self.log_evidences_native)
        return self._subhalo_array_from(values_native=self.stochastic_log_likelihoods_native)

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

        return stochastic_log_evidences

    @property
    def masses(self) -> List[float]:
        return [
        res.samples.median_pdf_instance.galaxies.subhalo.mass.mass_at_200
        for results in self.results_native
        for res in results
    ]

    @property
    def centres(self) -> List[Tuple[float]]:

        return [
            res.samples.median_pdf_instance.galaxies.subhalo.mass.centre
            for results in self.results_native
            for res in results
        ]
