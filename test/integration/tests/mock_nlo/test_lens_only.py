import autofit as af
from test.integration.tests.lens_only import lens_x1_galaxy
from test.integration.tests.lens_only import lens_x1_galaxy_hyper
from test.integration.tests.lens_only import runner


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


def run_a_mock(module):
    runner.run(
        module,
        test_name=f"{module.test_name}_mock",
        optimizer_class=MockNLO
    )


class TestCase:
    def test_lens_x1_galaxy(self):
        run_a_mock(lens_x1_galaxy)

    def test_lens_x1_galaxy_hyper(self):
        run_a_mock(lens_x1_galaxy_hyper)
