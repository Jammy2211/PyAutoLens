import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration.tests import runner

test_type = "model_mapper"
test_name = "constants_x2_galaxy"
data_type = "lens_only_dev_vaucouleurs"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class MMPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):
            self.galaxies.lens_0.light.axis_ratio = 0.2
            self.galaxies.lens_0.light.phi = 90.0
            self.galaxies.lens_0.light.centre_0 = 1.0
            self.galaxies.lens_0.light.centre_1 = 2.0
            self.galaxies.lens_1.light.axis_ratio = 0.2
            self.galaxies.lens_1.light.phi = 90.0
            self.galaxies.lens_1.light.centre_0 = 1.0
            self.galaxies.lens_1.light.centre_1 = 2.0

    phase1 = MMPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic),
            lens_1=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
