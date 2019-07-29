import autofit as af
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.integration.tests import runner

test_type = "lens_and_source_inversion"
test_name = "lens_both_x1_source_x1_rectangular"
data_type = "lens_light_and_source_smooth"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class SourcePix(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.mass.centre.centre_0 = 0.0
            self.galaxies.lens.mass.centre.centre_1 = 0.0
            self.galaxies.lens.mass.einstein_radius_in_units = 1.6
            self.galaxies.source.pixelization.shape_0 = 20.0
            self.galaxies.source.pixelization.shape_1 = 20.0

    phase1 = SourcePix(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.SphericalDevVaucouleurs,
                mass=mp.EllipticalIsothermal,
            ),
            source=gm.GalaxyModel(
                redshift=1.0, pixelization=pix.Rectangular, regularization=reg.Constant
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
