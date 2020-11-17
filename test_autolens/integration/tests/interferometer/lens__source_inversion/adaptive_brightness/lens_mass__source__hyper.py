import autofit as af
import autolens as al
from test_autolens.integration.tests.interferometer import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_brightness__hyper"
dataset_name = "mass_sie__source_sersic"
instrument = "sma"


def make_pipeline(name, path_prefix, real_space_mask):
    class Phase1(al.PhaseInterferometer):
        def customize_priors(self, results):
            self.galaxies.source.light.sersic_index = af.UniformPrior(3.9, 4.1)

    phase1 = Phase1(
        name="phase[1]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mp.EllipticalIsothermal, shear=al.mp.ExternalShear
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 50
    phase1.search.facc = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True,
        include_background_sky=True,
        include_background_noise=True,
    )
    phase2 = al.PhaseInterferometer(
        name="phase_2_weighted_regularization",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 30
    phase2.search.facc = 0.8

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True,
        include_background_sky=True,
        include_background_noise=True,
        include_inversion=True,
    )

    phase3 = al.PhaseInterferometer(
        name="phase[3]",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
                hyper_galaxy=phase2.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase2.result.hyper_combined.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase3.search.const_efficiency_mode = True
    phase3.search.n_live_points = 40
    phase3.search.facc = 0.8

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True,
        include_background_sky=True,
        include_background_noise=True,
        include_inversion=True,
    )

    return al.PipelineDataset(name, path_prefix, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
