import autofit as af
import autolens as al
from test.integration.tests import runner

test_type = "lens_only"
test_name = "lens_x2_light"
data_type = "lens_x2_light"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class LensPlanex2GalPhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens_0.light.centre_0 = -1.0
            self.galaxies.lens_0.light.centre_1 = -1.0
            self.galaxies.lens_1.light.centre_0 = 1.0
            self.galaxies.lens_1.light.centre_1 = 1.0

    def mask_function(image):
        return al.Mask.circular(
            shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=5.0
        )

    phase1 = LensPlanex2GalPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=al.GalaxyModel(
                redshift=0.5, light=al.light_profiles.EllipticalSersic
            ),
            lens_1=al.GalaxyModel(
                redshift=0.5, light=al.light_profiles.EllipticalSersic
            ),
        ),
        mask_function=mask_function,
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return al.PipelineImaging(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
