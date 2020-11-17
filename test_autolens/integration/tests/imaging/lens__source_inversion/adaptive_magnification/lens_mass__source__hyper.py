import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_magnification__hyper"
dataset_name = "mass_sie__source_sersic"
instrument = "vro"


def make_pipeline(name, path_prefix):

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    mass.centre.centre_0 = 0.0
    mass.centre.centre_1 = 0.0
    mass.einstein_radius = 1.6

    phase1 = al.PhaseImaging(
        name="phase[1]",
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=mass),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
            ),
        ),
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 60
    phase1.search.facc = 0.8

    phase1.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True, include_inversion=True
    )

    phase2 = al.PhaseImaging(
        name="phase[2]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase1.result.model.galaxies.source.pixelization,
                regularization=phase1.result.model.galaxies.source.regularization,
                hyper_galaxy=phase1.result.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 40
    phase2.search.facc = 0.8

    phase2.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True, include_inversion=True
    )

    phase3 = al.PhaseImaging(
        name="phase[3]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=af.last[-1].model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase1.result.model.galaxies.source.pixelization,
                regularization=phase1.result.model.galaxies.source.regularization,
                hyper_galaxy=phase2.result.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        search=search,
    )

    phase3.search.const_efficiency_mode = True
    phase3.search.n_live_points = 40
    phase3.search.facc = 0.8

    phase3.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True, include_inversion=True
    )

    return al.PipelineDataset(name, path_prefix, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
