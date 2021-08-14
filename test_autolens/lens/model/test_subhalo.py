import numpy as np

import autofit as af
import autolens as al
from autolens.lens.subhalo import SubhaloResult


class TestSubhaloResult:
    def test__result_derived_properties(self):

        lower_limit_lists = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]]

        grid_search_result = af.GridSearchResult(
            results=None,
            grid_priors=[
                af.UniformPrior(lower_limit=-2.0, upper_limit=2.0),
                af.UniformPrior(lower_limit=-3.0, upper_limit=3.0),
            ],
            lower_limits_lists=lower_limit_lists,
        )

        subhalo_result = SubhaloResult(
            grid_search_result=grid_search_result, result_no_subhalo=1
        )

        subhalo_array = subhalo_result._subhalo_array_from(
            values_native=np.array([[1.0, 2.0], [3.0, 4.0]])
        )

        assert isinstance(subhalo_array, al.Array2D)
        assert (subhalo_array.native == np.array([[3.0, 4.0], [1.0, 2.0]])).all()
