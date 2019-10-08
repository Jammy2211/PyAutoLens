from abc import ABC

import autofit as af


class Analysis(af.Analysis, ABC):
    def __init__(self, results=None):
        """
        An lens object

        Parameters
        ----------
        results: autofit.tools.pipeline.ResultsCollection
            The results of all previous phases
        """

        self.results = results

    @property
    def last_results(self):
        """
        Returns
        -------
        result: AbstractPhase.Result | None
            The result from the last phase
        """
        if self.results is not None:
            return self.results.last
