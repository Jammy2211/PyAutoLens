import autofit as af
import autolens as al
from test.integration.tests import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_brightness__hyper_constant_pass"
data_type = "lens_mass__source_smooth"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    phase1 = al.PhaseImaging(
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
            # Lens Mass, SIE -> SIE, Shear -> Shear #

            self.galaxies.lens = results.last.constant.galaxies.lens

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.galaxies.source.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.source.hyper_galaxy
            )

    phase2 = InversionPhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mass_profiles.EllipticalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pixelizations.VoronoiBrightnessImage,
                regularization=al.regularization.AdaptiveBrightness,
            ),
        ),
        inversion_pixel_limit=716,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 30
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=False, inversion=True
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):
            # Lens Mass, SIE -> SIE, Shear -> Shear #

            self.galaxies.lens = results.from_phase("phase_1").variable.galaxies.lens

            self.galaxies.source = results.from_phase(
                "phase_2"
            ).variable.galaxies.source

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
                redshift=0.5, mass=al.mass_profiles.EllipticalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pixelizations.VoronoiBrightnessImage,
                regularization=al.regularization.AdaptiveBrightness,
            ),
        ),
        inversion_pixel_limit=716,
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=False, inversion=True
    )

    return al.PipelineImaging(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
