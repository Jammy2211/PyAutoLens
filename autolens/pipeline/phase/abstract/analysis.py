from abc import ABC

import autofit as af
from autolens.lens import ray_tracing


class Analysis(af.Analysis, ABC):
    def __init__(self, cosmology, results=None):
        """
        An lens object

        Parameters
        ----------
        results: autofit.tools.pipeline.ResultsCollection
            The results of all previous phases
        """

        self.results = results
        self.cosmology = cosmology

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

    def tracer_for_instance(self, instance):
        return ray_tracing.Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
        )
