import autofit as af
import autolens as al
from test.integration.tests import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_brightness"
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

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase("phase_1").constant.galaxies.lens

    phase2 = InversionPhase(
        phase_name="phase_2_weighted_regularization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pixelizations.VoronoiMagnification,
                regularization=al.regularization.AdaptiveBrightness,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase("phase_1").variable.galaxies.lens

            self.galaxies.source = results.from_phase(
                "phase_2_weighted_regularization"
            ).constant.galaxies.source

    phase3 = InversionPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pixelizations.VoronoiMagnification,
                regularization=al.regularization.AdaptiveBrightness,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase("phase_3").constant.galaxies.lens

            self.galaxies.source = results.from_phase(
                "phase_2_weighted_regularization"
            ).variable.galaxies.source

    phase4 = InversionPhase(
        phase_name="phase_4_weighted_regularization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pixelizations.VoronoiMagnification,
                regularization=al.regularization.AdaptiveBrightness,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 40
    phase4.optimizer.sampling_efficiency = 0.8

    return al.PipelineImaging(name, phase1, phase2, phase3, phase4)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
