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

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8
    phase1.optimizer.evidence_tolerance = 10.0

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=phase1.result.instance.galaxies.lens,
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
    phase2.optimizer.evidence_tolerance = 1000.0

    phase2 = phase2.extend_with_multiple_hyper_phases(hyper_galaxy=True, inversion=True)

    phase3 = al.PhaseImaging(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=phase1.result.model.galaxies.lens,
            source=phase2.result.instance.galaxies.source,
        ),
        inversion_pixel_limit=50,
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8
    phase3.optimizer.evidence_tolerance = 1000.0

    phase3 = phase3.extend_with_multiple_hyper_phases(hyper_galaxy=True, inversion=True)

    return al.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
