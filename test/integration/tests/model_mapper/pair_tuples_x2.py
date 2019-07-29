import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration.tests import runner

test_type = "model_mapper"
test_name = "pair_tuples_x2"
data_type = "lens_only_dev_vaucouleurs"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class MMPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.light.centre_0 = self.galaxies.lens.light.axis_ratio
            self.galaxies.lens.light.centre_1 = self.galaxies.lens.light.intensity

    phase1 = MMPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic)),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
