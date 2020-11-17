import autofit as af
import autolens as al
from test_autolens.integration.tests.interferometer import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_brightness"
dataset_name = "mass_sie__source_sersic"
instrument = "sma"


def make_pipeline(name, path_prefix, real_space_mask):

    phase1 = al.PhaseInterferometer(
        name="phase[1]",
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase2 = al.PhaseInterferometer(
        name="phase_2_weighted_regularization",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
            ),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 40
    phase2.search.facc = 0.8

    phase3 = al.PhaseInterferometer(
        name="phase[3]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.model.galaxies.lens.mass,
                shear=phase1.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
            ),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase3.search.const_efficiency_mode = True
    phase3.search.n_live_points = 40
    phase3.search.facc = 0.8

    phase4 = al.PhaseInterferometer(
        name="phase_4_weighted_regularization",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.model.galaxies.source.pixelization,
                regularization=phase2.model.galaxies.source.pixelization,
            ),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase4.search.const_efficiency_mode = True
    phase4.search.n_live_points = 40
    phase4.search.facc = 0.8

    return al.PipelineDataset(name, path_prefix, phase1, phase2, phase3, phase4)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
