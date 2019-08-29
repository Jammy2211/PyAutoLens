import autofit as af
import autolens as al
from test.integration.tests import runner

test_type = "phase_features"
test_name = "positions__offset_centre"
data_type = "lens_sis__source_smooth__offset_centre"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    def mask_function(image):
        return al.Mask.circular(
            shape=image.shape,
            pixel_scale=image.pixel_scale,
            radius_arcsec=3.0,
            centre=(4.0, 4.0),
        )

    class LensPhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.mass.centre_0 = af.GaussianPrior(mean=4.0, sigma=0.1)
            self.galaxies.lens.mass.centre_1 = af.GaussianPrior(mean=4.0, sigma=0.1)
            self.galaxies.source.light.centre_0 = af.GaussianPrior(mean=4.0, sigma=0.1)
            self.galaxies.source.light.centre_1 = af.GaussianPrior(mean=4.0, sigma=0.1)

    phase1 = LensPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mass_profiles.SphericalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        mask_function=mask_function,
        positions_threshold=0.5,
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.8

    return al.PipelineImaging(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(
        sys.modules[__name__],
        positions=[[[5.6, 4.0], [4.0, 5.6], [2.4, 4.0], [4.0, 2.4]]],
    )
