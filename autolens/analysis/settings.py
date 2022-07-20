import copy
import numpy as np
from typing import Optional, Union

import autoarray as aa
import autogalaxy as ag

from autoconf import conf

from autolens.point.fit_point.max_separation import FitPositionsSourceMaxSeparation

from autolens import exc


class SettingsLens:
    def __init__(
        self,
        stochastic_likelihood_resamples: Optional[int] = None,
        stochastic_samples: int = 250,
        stochastic_histogram_bins: int = 10,
    ):

        self.stochastic_likelihood_resamples = stochastic_likelihood_resamples
        self.stochastic_samples = stochastic_samples
        self.stochastic_histogram_bins = stochastic_histogram_bins
