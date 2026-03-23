"""
Aggregator interface for dark-matter subhalo grid-search results.

``SubhaloAgg`` wraps a ``PyAutoFit`` ``GridSearchAggregator`` to provide a convenient
interface for loading and comparing the results of a subhalo detection grid search.

A subhalo grid search runs an independent non-linear search in each cell of an image-plane
grid, with the subhalo's (y, x) centre confined to that cell's region.  ``SubhaloAgg``
collates these per-cell results and exposes generators that yield ``FitImaging`` objects
for the best-fit model in each cell — both the "with subhalo" fit and (optionally) the
"no subhalo" baseline — enabling the per-cell log-evidence difference map to be computed.
"""
from typing import Optional

import autofit as af
import autoarray as aa

from autolens import exc


class SubhaloAgg:
    def __init__(
        self,
        aggregator_grid_search: af.GridSearchAggregator,
        settings: Optional[aa.Settings] = None,
    ):
        """
        Wraps a PyAutoFit aggregator in order to create generators of fits to imaging data, corresponding to the
        results of a non-linear search model-fit.
        """

        self.aggregator_grid_search = aggregator_grid_search
        self.settings = settings

        if len(aggregator_grid_search) == 0:
            raise exc.AggregatorException(
                "There is no grid search of results in the aggregator."
            )
        elif len(aggregator_grid_search) > 1:
            raise exc.AggregatorException(
                "There is more than one grid search of results in the aggregator - please filter the"
                "aggregator."
            )

    @property
    def grid_search_result(self) -> af.GridSearchResult:
        return self.aggregator_grid_search[0]["result"]
