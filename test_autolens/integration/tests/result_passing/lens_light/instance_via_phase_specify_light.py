import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "reult_passing"
test_name = "lens_light_instance_via_phase_specify_light"
data_label = "lens_sie__source_smooth"
instrument = "vro"


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.lp.SphericalDevVaucouleurs,
                mass=al.mp.EllipticalIsothermal,
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        sub_size=1,
        non_linear_class=non_linear_class,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 60
    phase1.search.sampling_efficiency = 0.8

    # This is an example of us passing results via phases, which we know will work.

    # We can be sure this works, because the paramete space of phase2 is (N = 12) and checking model.info shows the
    # lens light is passed as an instance.

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.instance.galaxies.lens.light,
                mass=phase1.result.model.galaxies.lens.mass,
            ),
            source=phase1.result.model.galaxies.source,
        ),
        sub_size=1,
        non_linear_class=non_linear_class,
    )

    return al.PipelineDataset(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
