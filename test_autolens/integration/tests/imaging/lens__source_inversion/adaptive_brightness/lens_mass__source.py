import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_brightness"
data_name = "lens_sie__source_smooth"
instrument = "hst"


def make_pipeline(name, folders, search=af.DynestyStatic()):

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 40
    phase1.search.facc = 0.8
    phase1.search.evidence_tolerance = 10.0

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=False,
        include_background_sky=False,
        include_background_noise=False,
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
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
            ),
        ),
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 40
    phase2.search.facc = 0.8
    phase2.search.evidence_tolerance = 10.0

    phase3 = al.PhaseImaging(
        phase_name="phase_3",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=phase1.model.galaxies.lens.mass),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
            ),
        ),
        search=search,
    )

    phase3.search.const_efficiency_mode = True
    phase3.search.n_live_points = 40
    phase3.search.facc = 0.8
    phase3.search.evidence_tolerance = 10.0

    phase4 = al.PhaseImaging(
        phase_name="phase_4_weighted_regularization",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase3.result.instance.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.model.galaxies.source.pixelization,
                regularization=phase2.model.galaxies.source.pixelization,
            ),
        ),
        search=search,
    )

    phase4.search.const_efficiency_mode = True
    phase4.search.n_live_points = 40
    phase4.search.facc = 0.8
    phase4.search.evidence_tolerance = 10.0

    return al.PipelineDataset(name, phase1, phase2, phase3, phase4)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
