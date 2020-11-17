import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "phase_features"
test_name = "positions__offset_centre"
dataset_name = "mass_sie__source_sersic__offset_centre"
instrument = "vro"


def make_pipeline(name, path_prefix):
    class LensPhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.mass.centre_0 = af.GaussianPrior(mean=4.0, sigma=0.1)
            self.galaxies.lens.mass.centre_1 = af.GaussianPrior(mean=4.0, sigma=0.1)
            self.galaxies.source.light.centre_0 = af.GaussianPrior(mean=4.0, sigma=0.1)
            self.galaxies.source.light.centre_1 = af.GaussianPrior(mean=4.0, sigma=0.1)

    phase1 = LensPhase(
        name="phase[1]",
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        positions_threshold=0.5,
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 30
    phase1.search.facc = 0.8

    return al.PipelineDataset(name, path_prefix, phase1)


if __name__ == "__main__":
    import sys

    runner.run(
        sys.modules[__name__],
        positions=[[(5.6, 4.0), (4.0, 5.6), (2.4, 4.0), (4.0, 2.4)]],
    )
