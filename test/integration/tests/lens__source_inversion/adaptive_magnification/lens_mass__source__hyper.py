import autofit as af
import autolens as al
from test.integration.tests import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_magnification__hyper"
data_type = "lens_mass__source_smooth"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class SourcePix(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.mass.centre.centre_0 = 0.0
            self.galaxies.lens.mass.centre.centre_1 = 0.0
            self.galaxies.lens.mass.einstein_radius = 1.6
            self.galaxies.source.pixelization.shape.shape_0 = 20.0
            self.galaxies.source.pixelization.shape.shape_1 = 20.0

    phase1 = SourcePix(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mass_profiles.EllipticalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pixelizations.VoronoiMagnification,
                regularization=al.regularization.Constant,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    phase1.extend_with_multiple_hyper_phases(hyper_galaxy=True)

    class HyperLensSourcePlanePhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.hyper_galaxy = results.from_phase(
                "phase_1"
            ).hyper_combined.constant.galaxies.lens.hyper_galaxy

            self.galaxies.source.hyper_galaxy = results.from_phase(
                "phase_1"
            ).hyper_combined.constant.galaxies.source.hyper_galaxy

    phase2 = HyperLensSourcePlanePhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.result.variable.galaxies.lens.mass,
                hyper_galaxy=al.HyperGalaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase1.result.variable.galaxies.source.pixelization,
                regularization=phase1.result.variable.galaxies.source.regularization,
                hyper_galaxy=al.HyperGalaxy,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    return al.PipelineImaging(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
