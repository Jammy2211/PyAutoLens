import autofit as af
import autolens as al
from test_autolens.integration.tests.interferometer import runner

test_type = "lens__source"
test_name = "light_sersic___mass__source"
dataset_name = "mass_sie__source_sersic"
instrument = "sma"


def make_pipeline(name, path_prefix, real_space_mask):

    phase1 = al.PhaseInterferometer(
        name="phase[1]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.lp.SphericalDevVaucouleurs,
                mass=al.mp.EllipticalIsothermal,
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 60
    phase1.search.facc = 0.8

    return al.PipelineDataset(name, path_prefix, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
