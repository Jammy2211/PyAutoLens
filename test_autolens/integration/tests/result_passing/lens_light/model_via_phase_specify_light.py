import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "reult_passing"
test_name = "lens_light_model_via_phase_specify_light"
data_name = "lens_sie__source_smooth"
instrument = "vro"


def make_pipeline(name, path_prefix, search=af.DynestyStatic()):

    phase1 = al.PhaseImaging(
        name="phase_1",
        path_prefix=path_prefix,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.lp.SphericalDevVaucouleurs,
                mass=al.mp.EllipticalIsothermal,
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        sub_size=1,
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 60
    phase1.search.facc = 0.8

    # This is an example of us passing results via phases, which we know will work.

    # We can be sure this works, because the paramete space of phase2 is the same as phase1 (N = 16).

    phase2 = al.PhaseImaging(
        name="phase_2",
        path_prefix=path_prefix,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.model.galaxies.lens.light,
                mass=phase1.result.model.galaxies.lens.mass,
            ),
            source=phase1.result.model.galaxies.source,
        ),
        sub_size=1,
        search=search,
    )

    return al.PipelineDataset(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
