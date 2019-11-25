import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "regression"
test_name = "new_api_pixelization"
data_type = "lens_mass__source_smooth"
data_resolution = "lsst"


def make_pipeline(
    name,
    phase_folders,
    pipeline_pixelization=al.pix.VoronoiBrightnessImage,
    pipeline_regularization=al.reg.AdaptiveBrightness,
    optimizer_class=af.MultiNest,
):
    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            source=al.GalaxyModel(redshift=1.0, light=al.lp.SphericalExponential),
            lens=al.GalaxyModel(redshift=0.5, light=al.mp.SphericalIsothermal()),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    phase6 = al.PhaseImaging(
        phase_name="phase_6_initialize_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=al.mp.SphericalIsothermal(),
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_pixelization,
                regularization=pipeline_regularization,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=optimizer_class,
    )

    phase6.optimizer.const_efficiency_mode = True
    phase6.optimizer.n_live_points = 20
    phase6.optimizer.sampling_efficiency = 0.8

    phase6 = phase6.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    phase7 = al.PhaseImaging(
        phase_name="phase_7__lens_sersic_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=al.mp.SphericalExponential(),
                hyper_galaxy=phase6.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase6.result.instance.galaxies.source.pixelization,
                regularization=phase6.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase6.result.hyper_combined.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=optimizer_class,
    )

    phase7.optimizer.const_efficiency_mode = True
    phase7.optimizer.n_live_points = 75
    phase7.optimizer.sampling_efficiency = 0.2

    phase7 = phase7.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    return al.PipelineDataset(name, phase1, phase6, phase7, hyper_mode=False)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
