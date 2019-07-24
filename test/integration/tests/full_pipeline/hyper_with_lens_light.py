import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline import pipeline as pl
from autolens.pipeline.phase import phase_imaging
from test.integration.tests import runner

test_type = "full_pipeline"
test_name = "hyper_with_lens_light"
data_resolution = "LSST"


def make_pipeline(        
        name,
        phase_folders,
        optimizer_class=af.MultiNest):
    
    phase1 = phase_imaging.LensPlanePhase(
        phase_name="phase_1_lens_sersic",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic)
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
    )

    class LensSubtractedPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):

            ## Lens Light Sersic -> Sersic ##

            self.lens_galaxies.lens.light = results.from_phase(
                "phase_1_lens_sersic"
            ).constant.lens_galaxies.lens.light

            ## Lens Mass, Move centre priors to centre of lens light ###

            self.lens_galaxies.lens.mass.centre = (
                results.from_phase("phase_1_lens_sersic")
                .variable_absolute(a=0.1)
                .lens_galaxies.lens.light.centre
            )

            ## Set all hyper-galaxies if feature is turned on ##

            self.lens_galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.lens_galaxies.lens.hyper_galaxy
            )

            self.hyper_image_sky = (
                results.last.hyper_combined.constant.hyper_image_sky
            )

            self.hyper_noise_background = (
                results.last.hyper_combined.constant.hyper_noise_background
            )

    phase2 = LensSubtractedPhase(
        phase_name="phase_2_lens_sie_shear_source_sersic",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            )
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic)
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
    )

    class LensSourcePhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):

            ## Lens Light, Sersic -> Sersic ###

            self.lens_galaxies.lens.light = results.from_phase(
                "phase_1_lens_sersic"
            ).variable.lens_galaxies.lens.light

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens.mass = results.from_phase(
                "phase_2_lens_sie_shear_source_sersic"
            ).variable.lens_galaxies.lens.mass

            self.lens_galaxies.lens.shear = results.from_phase(
                "phase_2_lens_sie_shear_source_sersic"
            ).variable.lens_galaxies.lens.shear

            ### Source Light, Sersic -> Sersic ###

            self.source_galaxies.source = results.from_phase(
                "phase_2_lens_sie_shear_source_sersic"
            ).variable.source_galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            self.lens_galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.lens_galaxies.lens.hyper_galaxy
            )

            self.hyper_image_sky = (
                results.last.hyper_combined.constant.hyper_image_sky
            )

            self.hyper_noise_background = (
                results.last.hyper_combined.constant.hyper_noise_background
            )

    phase3 = LensSourcePhase(
        phase_name="phase_3_lens_sersic_sie_shear_source_sersic",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            )
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic)
        ),
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
    )

    return pl.PipelineImaging(name, phase1, phase2, phase3, hyper_mode=True)


if __name__ == "__main__":
    import sys

    runner.run(
        sys.modules[__name__]
    )
