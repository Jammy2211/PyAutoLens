import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration.tests import runner

test_type = "lens_only"
test_name = "lens_x2_galaxies_separate"
data_type = "lens_only_x2_galaxies"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    def modify_mask_function(image):
        return msk.Mask.circular(
            shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=5.0
        )

    class LensPlaneGalaxy0Phase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens_0.sersic.centre_0 = -1.0
            self.galaxies.lens_0.sersic.centre_1 = -1.0

    phase1 = LensPlaneGalaxy0Phase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(lens_0=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic)),
        mask_function=modify_mask_function,
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class LensPlaneGalaxy1Phase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens_0 = results.from_phase(
                "phase_1"
            ).constant.galaxies.lens_0

            self.galaxies.lens_1.sersic.centre_0 = 1.0
            self.galaxies.lens_1.sersic.centre_1 = 1.0

    phase2 = LensPlaneGalaxy1Phase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic),
            lens_1=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic),
        ),
        mask_function=modify_mask_function,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    class LensPlaneBothGalaxyPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens_0 = results.from_phase(
                "phase_1"
            ).variable.galaxies.lens_0

            self.galaxies.lens_1 = results.from_phase(
                "phase_2"
            ).variable.galaxies.lens_0

            self.galaxies.lens_0.sersic.centre_0 = -1.0
            self.galaxies.lens_0.sersic.centre_1 = -1.0
            self.galaxies.lens_1.sersic.centre_0 = 1.0
            self.galaxies.lens_1.sersic.centre_1 = 1.0

    phase3 = LensPlaneBothGalaxyPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic),
            lens_1=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic),
        ),
        mask_function=modify_mask_function,
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
