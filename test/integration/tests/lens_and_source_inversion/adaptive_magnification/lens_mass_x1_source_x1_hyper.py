import autofit as af
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline import pipeline as pl
from autolens.pipeline.phase import phase_imaging
from test.integration.tests import runner

test_type = "lens_and_source_inversion"
test_name = "lens_mass_x1_source_x1_adaptive_magnification_hyper"
data_type = "no_lens_light_and_source_smooth"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class SourcePix(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.mass.centre.centre_0 = 0.0
            self.galaxies.lens.mass.centre.centre_1 = 0.0
            self.galaxies.lens.mass.einstein_radius = 1.6
            self.galaxies.source.pixelization.shape.shape_0 = 20.0
            self.galaxies.source.pixelization.shape.shape_1 = 20.0

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

    phase1.extend_with_multiple_hyper_phases(hyper_galaxy=True)

    class HyperLensSourcePlanePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.hyper_galaxy = results.from_phase(
                "phase_1"
            ).hyper_combined.constant.galaxies.lens.hyper_galaxy

            self.galaxies.lens.mass = results.from_phase(
                "phase_1"
            ).variable.galaxies.lens.mass

            self.galaxies.source.hyper_galaxy = results.from_phase(
                "phase_1"
            ).hyper_combined.constant.galaxies.source.hyper_galaxy

            self.galaxies.source.pixelization = results.from_phase(
                "phase_1"
            ).variable.galaxies.source.pixelization

            self.galaxies.source.regularization = results.from_phase(
                "phase_1"
            ).variable.galaxies.source.regularization

    phase2 = HyperLensSourcePlanePhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5, mass=mp.EllipticalIsothermal, hyper_galaxy=g.HyperGalaxy
            ),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant,
                hyper_galaxy=g.HyperGalaxy,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
