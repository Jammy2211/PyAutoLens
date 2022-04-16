import autoarray as aa

from autolens.analysis.result import Result


class ResultPoint(Result):
    @property
    def grid(self):
        return aa.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

    @property
    def max_log_likelihood_fit(self):

        return self.analysis.fit_positions_for(instance=self.instance)
