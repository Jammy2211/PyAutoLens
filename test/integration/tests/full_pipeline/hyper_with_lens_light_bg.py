import autofit as af
import autolens as al
from test.integration.tests import runner

test_type = "full_pipeline"
test_name = "hyper_with_lens_light_bg"
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
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.light_profiles.EllipticalSersic)
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    class LensSubtractedPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Light Sersic -> Sersic ##

            self.galaxies.lens.light = results.from_phase(
                "phase_1__lens_sersic"
            ).constant.galaxies.lens.light

            ## Lens Mass, Move centre priors to centre of lens light ###

            self.galaxies.lens.mass.centre = (
                results.from_phase("phase_1__lens_sersic")
                .variable_absolute(a=0.1)
                .galaxies.lens.light.centre
            )

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase2 = LensSubtractedPhase(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    class LensSourcePhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Light, Sersic -> Sersic ###

            self.galaxies.lens.light = results.from_phase(
                "phase_1__lens_sersic"
            ).variable.galaxies.lens.light

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass

            self.galaxies.lens.shear = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.lens.shear

            ### Source Light, Sersic -> Sersic ###

            self.galaxies.source = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.source

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase3 = LensSourcePhase(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.light = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).constant.galaxies.lens.light

            self.galaxies.lens.mass = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).constant.galaxies.lens.mass

            self.galaxies.lens.shear = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).constant.galaxies.lens.shear

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase4 = InversionPhase(
        phase_name="phase_4__initialize_magnification_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
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

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=False,
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source.pixelization = results.from_phase(
                "phase_4__initialize_magnification_inversion"
            ).constant.galaxies.source.pixelization

            self.galaxies.source.regularization = results.from_phase(
                "phase_4__initialize_magnification_inversion"
            ).constant.galaxies.source.regularization

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase5 = InversionPhase(
        phase_name="phase_5__lens_sersic_sie__source_magnification_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
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

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 75
    phase5.optimizer.sampling_efficiency = 0.2

    phase5 = phase5.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=False,
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.light = results.from_phase(
                "phase_5__lens_sersic_sie__source_magnification_inversion"
            ).constant.galaxies.lens.light

            self.galaxies.lens.mass = results.from_phase(
                "phase_5__lens_sersic_sie__source_magnification_inversion"
            ).constant.galaxies.lens.mass

            self.galaxies.lens.shear = results.from_phase(
                "phase_5__lens_sersic_sie__source_magnification_inversion"
            ).constant.galaxies.lens.shear

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase6 = InversionPhase(
        phase_name="phase_6_initialize_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
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

    phase6.optimizer.const_efficiency_mode = True
    phase6.optimizer.n_live_points = 20
    phase6.optimizer.sampling_efficiency = 0.8

    phase6 = phase6.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_5__lens_sersic_sie__source_magnification_inversion"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source.pixelization = results.from_phase(
                "phase_6_initialize_inversion"
            ).hyper_combined.constant.galaxies.source.pixelization

            self.galaxies.source.regularization = results.from_phase(
                "phase_6_initialize_inversion"
            ).hyper_combined.constant.galaxies.source.regularization

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.galaxies.source.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.source.hyper_galaxy
            )

            self.hyper_image_sky = results.last.hyper_combined.constant.hyper_image_sky

            self.hyper_background_noise = (
                results.last.hyper_combined.constant.hyper_background_noise
            )

    phase7 = InversionPhase(
        phase_name="phase_7__lens_sersic_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
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

    phase7.optimizer.const_efficiency_mode = True
    phase7.optimizer.n_live_points = 75
    phase7.optimizer.sampling_efficiency = 0.2

    phase7 = phase7.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    return al.PipelineImaging(
        name, phase1, phase2, phase3, phase4, phase5, phase6, phase7, hyper_mode=False
    )


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
