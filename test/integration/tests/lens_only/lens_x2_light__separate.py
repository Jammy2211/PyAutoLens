import autofit as af
import autolens as al
from test.integration.tests import runner

test_type = "lens_only"
test_name = "lens_x2_light__separate"
data_type = "lens_x2_light"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    def modify_mask_function(image):
        return al.Mask.circular(
            shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=5.0
        )

    class LensPlaneGalaxy0Phase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens_0.sersic.centre_0 = -1.0
            self.galaxies.lens_0.sersic.centre_1 = -1.0

    phase1 = LensPlaneGalaxy0Phase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=al.GalaxyModel(
                redshift=0.5, sersic=al.light_profiles.EllipticalSersic
            )
        ),
        mask_function=modify_mask_function,
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class LensPlaneGalaxy1Phase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens_0 = results.from_phase(
                "phase_1"
            ).constant.galaxies.lens_0

            self.galaxies.lens_1.sersic.centre_0 = 1.0
            self.galaxies.lens_1.sersic.centre_1 = 1.0

    phase2 = LensPlaneGalaxy1Phase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=al.GalaxyModel(
                redshift=0.5, sersic=al.light_profiles.EllipticalSersic
            ),
            lens_1=al.GalaxyModel(
                redshift=0.5, sersic=al.light_profiles.EllipticalSersic
            ),
        ),
        mask_function=modify_mask_function,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    class LensPlaneBothGalaxyPhase(al.PhaseImaging):
        def customize_priors(self, results):

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
            lens_0=al.GalaxyModel(
                redshift=0.5, sersic=al.light_profiles.EllipticalSersic
            ),
            lens_1=al.GalaxyModel(
                redshift=0.5, sersic=al.light_profiles.EllipticalSersic
            ),
        ),
        mask_function=modify_mask_function,
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return al.PipelineImaging(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
