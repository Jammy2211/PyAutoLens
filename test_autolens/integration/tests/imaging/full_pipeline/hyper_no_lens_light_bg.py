import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "full_pipeline"
test_name = "hyper_no_lens_light_bg"
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
        phase_name="phase_1__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mp.EllipticalIsothermal, shear=al.mp.ExternalShear
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2_initialize_magnification_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.Rectangular,
                regularization=al.reg.Constant,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sie__source_magnification_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=af.last[-1].model.galaxies.lens.mass,
                shear=af.last[-1].model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.result.model.galaxies.source.pixelization,
                regularization=phase2.result.model.galaxies.source.regularization,
            ),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    phase4 = al.PhaseImaging(
        phase_name="phase_4__initialize_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_pixelization,
                regularization=pipeline_regularization,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=optimizer_class,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    phase5 = al.PhaseImaging(
        phase_name="phase_5__lens_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase3.result.model.galaxies.lens.mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase4.result.hyper_combined.instance.galaxies.source.pixelization,
                regularization=phase4.result.hyper_combined.instance.galaxies.source.regularization,
                hyper_galaxy=phase4.result.hyper_combined.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase4.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase4.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=optimizer_class,
    )

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 50
    phase5.optimizer.sampling_efficiency = 0.5

    phase5 = phase5.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    return al.PipelineDataset(
        name, phase1, phase2, phase3, phase4, phase5, hyper_mode=False
    )


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
