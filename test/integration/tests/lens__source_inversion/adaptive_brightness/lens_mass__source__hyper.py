import autofit as af
import autolens as al
from test.integration.tests import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_brightness__hyper"
data_type = "lens_mass__source_smooth"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class Phase1(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.source.light.sersic_index = af.UniformPrior(3.9, 4.1)

    phase1 = Phase1(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mass_profiles.EllipticalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 50
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(hyper_galaxy=True)

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.galaxies.source.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.source.hyper_galaxy
            )

    phase2 = InversionPhase(
        phase_name="phase_2_weighted_regularization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.constant.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pixelizations.VoronoiBrightnessImage,
                regularization=al.regularization.AdaptiveBrightness,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 30
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_multiple_hyper_phases(hyper_galaxy=True, inversion=True)

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.galaxies.source.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.source.hyper_galaxy
            )

    phase3 = InversionPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.result.variable.galaxies.lens.mass,
                shear=phase1.result.variable.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.result.constant.galaxies.source.pixelization,
                regularization=phase2.result.constant.galaxies.source.regularization,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8

    phase3 = phase3.extend_with_multiple_hyper_phases(hyper_galaxy=True, inversion=True)

    return al.PipelineImaging(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
