import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_brightness"
data_name = "lens_sie__source_smooth"
instrument = "euclid"


def make_pipeline(name, folders, search=af.DynestyStatic()):

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=af.DynestyStatic(n_live_points=40, evidence_tolerance=10.0),
    )

    pixeliation = af.PriorModel(al.pix.VoronoiBrightnessImage)
    pixeliation.pixels = 100

    phase1 = phase1.extend_with_multiple_hyper_phases(
        setup=al.PipelineSetup(), include_inversion=False
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        folders=folders,
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
        setup=al.PipelineSetup(), include_inversion=True
    )

    phase3 = al.PhaseImaging(
        phase_name="phase_3",
        folders=folders,
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

    return al.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
