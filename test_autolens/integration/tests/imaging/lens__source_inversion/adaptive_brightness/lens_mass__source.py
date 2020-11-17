import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_brightness"
dataset_name = "mass_sie__source_sersic"
instrument = "euclid"


def make_pipeline(name, path_prefix):

    phase1 = al.PhaseImaging(
        name="phase[1]",
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=af.DynestyStatic(n_live_points=40, evidence_tolerance=10.0),
    )

    pixeliation = af.PriorModel(al.pix.VoronoiBrightnessImage)
    pixeliation.pixels = 100

    phase1 = phase1.extend_with_multiple_hyper_phases(
        setup_hyper=al.SetupPipeline(), include_inversion=False
    )

    phase2 = al.PhaseImaging(
        name="phase[2]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.instance.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=pixeliation,
                regularization=al.reg.AdaptiveBrightness,
            ),
        ),
        search=af.DynestyStatic(n_live_points=40, evidence_tolerance=10.0),
    )

    phase2 = phase2.extend_with_multiple_hyper_phases(
        setup_hyper=al.SetupPipeline(), include_inversion=True
    )

    phase3 = al.PhaseImaging(
        name="phase[3]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
            ),
        ),
        search=af.DynestyStatic(n_live_points=40, evidence_tolerance=10.0),
    )

    return al.PipelineDataset(name, path_prefix, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
