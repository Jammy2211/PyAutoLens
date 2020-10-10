import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source"
test_name = "lens_light_mass__source"
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
        search=search,
    )

    return al.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
