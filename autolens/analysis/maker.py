import autogalaxy as ag

from autolens.analysis.preloads import Preloads


class FitMaker(ag.FitMaker):
    @property
    def preloads_cls(self):
        return Preloads
