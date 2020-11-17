from autoconf import conf
import autoarray as aa
import numpy as np
from autogalaxy.galaxy import galaxy as g
from autolens.pipeline.phase import dataset


class Result(dataset.Result):
    @property
    def max_log_likelihood_fit(self):

        return self.analysis.positions_fit_for_tracer(
            tracer=self.max_log_likelihood_tracer
        )
