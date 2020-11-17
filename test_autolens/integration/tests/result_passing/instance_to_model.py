import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "reult_passing"
test_name = "instance_to_model"
dataset_name = "mass_sie__source_sersic"
instrument = "vro"


def make_pipeline(name, path_prefix):

    light = af.PriorModel(al.lp.SphericalDevVaucouleurs)

    light.effective_radius = 1.0

    phase1 = al.PhaseImaging(
        name="phase[1]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, light=light, mass=al.mp.EllipticalIsothermal
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        search=search,
    )

    # This is an example of us passing results via phases, which we know will work.

    # We can be sure this works, because the paramete space of phase2 is (N = 12) and checking model.info shows the
    # lens light is passed as an instance.

    light = af.PriorModel(al.lp.SphericalDevVaucouleurs)

    light.effective_radius = af.GaussianPrior(
        mean=phase1.result.model.galaxies.lens.light.effective_radius, sigma=0.1
    )

    phase2 = al.PhaseImaging(
        name="phase[2]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, light=light, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=phase1.result.model.galaxies.source,
        ),
        search=search,
    )

    return al.PipelineDataset(name, path_prefix, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
