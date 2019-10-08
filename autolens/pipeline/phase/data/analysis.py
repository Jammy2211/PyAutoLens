from abc import ABC

from autolens import exc
from autolens.lens import lens_fit
from autolens.pipeline.phase import abstract


class Analysis(abstract.analysis.Analysis, ABC):
    @property
    def lens_data(self):
        raise NotImplementedError()
