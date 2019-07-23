import autofit as af
from test.integration.tests.lens_only import lens_x1_galaxy


class MockNLO(af.NonLinearOptimizer):
    def fit(self, analysis):
        instance = self.variable.instance_from_prior_medians()
        fit = analysis.fit(
            instance
        )
        return af.Result(
            instance,
            fit,
            self.variable,
            gaussian_tuples=[
                (
                    prior.mean,
                    prior.width
                )
                for prior
                in self.variable.priors
            ]
        )


def test_lens_x1_galaxy():
    lens_x1_galaxy.pipeline(
        MockNLO
    )
