import autofit as af
import autolens as al
from test.integration.tests import runner

test_type = "full_pipeline"
test_name = "hyper_no_lens_light_bg"
data_type = "lens_mass__source_smooth"
data_resolution = "LSST"


def make_pipeline(
    name,
    phase_folders,
    pipeline_pixelization=al.pixelizations.VoronoiBrightnessImage,
    pipeline_regularization=al.regularization.AdaptiveBrightness,
    optimizer_class=af.MultiNest,
):

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.lens

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase2 = InversionPhase(
        phase_name="phase_1_initialize_magnification_inversion",
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
                regularization=al.regularization.Constant,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=False,
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ### Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_1_initialize_magnification_inversion"
            ).variable.galaxies.source

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase3 = InversionPhase(
        phase_name="phase_3__lens_sie__source_magnification_inversion",
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
                regularization=al.regularization.Constant,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=False,
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_3__lens_sie__source_magnification_inversion"
            ).variable.galaxies.lens

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase4 = InversionPhase(
        phase_name="phase_4__initialize_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_pixelization,
                regularization=pipeline_regularization,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ### Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_3__lens_sie__source_magnification_inversion"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_4__initialize_inversion"
            ).hyper_combined.variable.galaxies.source

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.galaxies.source.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.source.hyper_galaxy
            )

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase5 = InversionPhase(
        phase_name="phase_5__lens_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_pixelization,
                regularization=pipeline_regularization,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 50
    phase5.optimizer.sampling_efficiency = 0.5

    phase5 = phase5.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    return al.PipelineImaging(
        name, phase1, phase2, phase3, phase4, phase5, hyper_mode=False
    )


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
