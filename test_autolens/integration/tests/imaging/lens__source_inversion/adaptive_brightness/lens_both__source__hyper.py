import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source_inversion"
test_name = "lens_both__source_adaptive_brightness__hyper"
dataset_name = "light_sersic__source_sersic"
instrument = "vro"


def make_pipeline(name, path_prefix):

    phase1 = al.PhaseImaging(
        name="phase[1]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.lp.EllipticalSersic,
                mass=al.mp.EllipticalIsothermal,
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 40
    phase1.search.facc = 0.8
    phase1.search.evidence_tolerance = 10.0

    phase2 = al.PhaseImaging(
        name="phase[2]",
        galaxies=dict(
            lens=phase1.result.instance.galaxies.lens,
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
    phase2.search.evidence_tolerance = 1000.0

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True, include_inversion=True
    )

    phase3 = al.PhaseImaging(
        name="phase[3]",
        galaxies=dict(
            lens=phase1.result.model.galaxies.lens,
            source=phase2.result.instance.galaxies.source,
        ),
        search=search,
    )

    phase3.search.const_efficiency_mode = True
    phase3.search.n_live_points = 40
    phase3.search.facc = 0.8
    phase3.search.evidence_tolerance = 1000.0

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True, include_inversion=True
    )

    return al.PipelineDataset(name, path_prefix, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
