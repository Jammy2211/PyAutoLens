import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from test.integration.tests import runner

test_type = "phase_features"
test_name = "positions_mass_x1_source_x1_offset_centre"
data_type = "no_lens_light_spherical_mass_and_source_smooth_offset_centre"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    def mask_function(image):
        return msk.Mask.circular(
            shape=image.shape,
            pixel_scale=image.pixel_scale,
            radius_arcsec=3.0,
            centre=(4.0, 4.0),
        )

    class LensPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.mass.centre_0 = af.GaussianPrior(mean=4.0, sigma=0.1)
            self.galaxies.lens.mass.centre_1 = af.GaussianPrior(mean=4.0, sigma=0.1)
            self.galaxies.source.light.centre_0 = af.GaussianPrior(mean=4.0, sigma=0.1)
            self.galaxies.source.light.centre_1 = af.GaussianPrior(mean=4.0, sigma=0.1)

    phase1 = LensPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.SphericalIsothermal),
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic),
        ),
        mask_function=mask_function,
        positions_threshold=0.5,
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(
        sys.modules[__name__],
        positions=[[[5.6, 4.0], [4.0, 5.6], [2.4, 4.0], [4.0, 2.4]]],
    )
