import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import mass_profiles as mp
from test.integration.tests import runner

test_type = "lens_and_source_inversion"
test_name = "lens_mass_x1_source_x1_adaptive_magnification"
data_type = "no_lens_light_and_source_smooth"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class SourcePix(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.mass.centre.centre_0 = 0.0
            self.galaxies.lens.mass.centre.centre_1 = 0.0
            self.galaxies.lens.mass.einstein_radius = 1.6

    phase1 = SourcePix(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_inversion_phase()

    class SourcePix(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens = results.from_phase("phase_1").variable.galaxies.lens

            self.galaxies.source = results.last.inversion.constant.galaxies.source

    phase2 = SourcePix(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_inversion_phase()

    return pl.PipelineImaging(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
