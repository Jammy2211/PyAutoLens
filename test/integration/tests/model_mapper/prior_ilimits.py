import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration.tests import runner

test_type = "model_mapper"
test_name = "prior_limits"
data_type = "lens_only_dev_vaucouleurs"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class MMPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.sersic.centre_0 = 0.0
            self.galaxies.lens.sersic.centre_1 = 0.0
            self.galaxies.lens.sersic.axis_ratio = af.UniformPrior(
                lower_limit=-0.5, upper_limit=0.1
            )
            self.galaxies.lens.sersic.phi = 90.0
            self.galaxies.lens.sersic.intensity = af.UniformPrior(
                lower_limit=-0.5, upper_limit=0.1
            )
            self.galaxies.lens.sersic.effective_radius = 1.3
            self.galaxies.lens.sersic.sersic_index = 3.0

    phase1 = MMPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic)),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    class MMPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.sersic.intensity = results.from_phase(
                "phase_1"
            ).variable.lens.sersic.intensity
            self.galaxies.lens = results.from_phase("phase_1").variable.lens

    phase2 = MMPhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic)),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
