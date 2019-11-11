import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source_inversion"
test_name = "lens_both__source_adaptive_brightness__hyper"
data_type = "lens_light__source_smooth"
data_resolution = "lsst"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.lp.EllipticalSersic,
                mass=al.mp.EllipticalIsothermal,
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        optimizer_class=optimizer_class,
    )

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.light = results.from_phase(
                "phase_1"
            ).instance.galaxies.lens.light
            self.galaxies.lens.mass = results.from_phase(
                "phase_1"
            ).instance.galaxies.lens.mass

    phase2 = InversionPhase(
        phase_name="phase_2_weighted_regularization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.lp.EllipticalSersic,
                mass=al.mp.EllipticalIsothermal,
                shear=al.mp.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
            ),
        ),
        inversion_pixel_limit=50,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_multiple_hyper_phases(hyper_galaxy=True, inversion=True)

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase("phase_1").model.galaxies.lens

            self.galaxies.source.pixelization = (
                results.last.inversion.instance.galaxies.source.pixelization
            )
            self.galaxies.source.regularization = (
                results.last.inversion.instance.galaxies.source.regularization
            )

    phase3 = InversionPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.lp.EllipticalSersic,
                mass=al.mp.EllipticalIsothermal,
                shear=al.mp.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
            ),
        ),
        inversion_pixel_limit=50,
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8

    phase3 = phase3.extend_with_multiple_hyper_phases(hyper_galaxy=True, inversion=True)

    return al.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
