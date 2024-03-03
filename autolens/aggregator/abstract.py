from __future__ import annotations
from abc import ABC
from typing import Generator

import autofit as af


class AbstractAgg(ABC):

    def max_log_likelihood_gen_from(self) -> Generator:
        """
        Returns a generator using the maximum likelihood instance of a non-linear search.

        This generator creates a list containing the maximum log instance of every result loaded in the aggregator.

        For example, in **PyAutoLens**, by overwriting the `make_gen_from` method this returns a generator
        of `Plane` objects from a PyAutoFit aggregator. This generator then generates a list of the maximum log
        likelihood `Plane` objects for all aggregator results.
        """

        def func_gen(fit: af.Fit) -> Generator:

            return self.object_via_gen_from(fit=fit)

        return self.aggregator.map(func=func_gen)
