import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline import pipeline as pl
from autolens.pipeline.phase import phase_imaging
from test.integration.tests import runner

test_type = "lens_and_source_inversion"
test_name = "lens_mass_x1_source_x1_adaptive_weighted_hyper_constant_pass"
data_type = "no_lens_light_and_source_smooth"
data_resolution = "LSST"


def make_pipeline(
        name,
        phase_folders,
        optimizer_class=af.MultiNest
):
    phase1 = phase_imaging.LensSourcePlanePhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic)
        ),
        optimizer_class=optimizer_class
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 50
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(hyper_galaxy=True)

    class InversionPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):
            # Lens Mass, SIE -> SIE, Shear -> Shear #

            self.lens_galaxies.lens = results.last.constant.lens_galaxies.lens

            self.lens_galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.lens_galaxies.lens.hyper_galaxy
            )

            self.source_galaxies.source.hyper_galaxy = (
                results.last.hyper_combined.constant.source_galaxies.source.hyper_galaxy
            )

    phase2 = InversionPhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiBrightnessImage,
                regularization=reg.AdaptiveBrightness,
            )
        ),
        inversion_pixel_limit=500,
        optimizer_class=optimizer_class
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 30
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        inversion=True
    )

    class InversionPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):
            # Lens Mass, SIE -> SIE, Shear -> Shear #

            self.lens_galaxies.lens = results.from_phase(
                "phase_1"
            ).variable.lens_galaxies.lens

            self.source_galaxies = results.last.hyper_combined.constant.source_galaxies

            self.lens_galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.lens_galaxies.lens.hyper_galaxy
            )

            self.source_galaxies.source.hyper_galaxy = (
                results.last.hyper_combined.constant.source_galaxies.source.hyper_galaxy
            )

    phase3 = InversionPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiBrightnessImage,
                regularization=reg.AdaptiveBrightness,
            )
        ),
        inversion_pixel_limit=500,
        optimizer_class=optimizer_class
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        inversion=True
    )

    return pl.PipelineImaging(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(
        sys.modules[__name__]
    )
