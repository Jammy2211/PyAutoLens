import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "full_pipeline"
test_name = "hyper_with_lens_light_bg"
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
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    #    mass.centre = phase1.result.model_absolute(a=0.1).galaxies.lens.bulge.centre
    mass.centre = phase1.result.model.galaxies.lens.light.centre

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.instance.galaxies.lens.light,
                mass=mass,
                shear=al.mp.ExternalShear,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.model.galaxies.lens.light,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
                hyper_galaxy=phase2.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=phase2.result.model.galaxies.source.light
            ),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    phase4 = al.PhaseImaging(
        phase_name="phase_4__initialize_magnification_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase3.result.instance.galaxies.lens.light,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase3.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
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
        inversion=False,
    )

    phase5 = al.PhaseImaging(
        phase_name="phase_5__lens_sersic_sie__source_magnification_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase3.result.model.galaxies.lens.light,
                mass=phase3.result.model.galaxies.lens.mass,
                shear=phase3.result.model.galaxies.lens.shear,
                hyper_galaxy=phase4.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase4.result.instance.galaxies.source.pixelization,
                regularization=phase4.result.instance.galaxies.source.regularization,
            ),
        ),
        hyper_image_sky=phase4.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase4.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=optimizer_class,
    )

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 75
    phase5.optimizer.sampling_efficiency = 0.2

    phase5 = phase5.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=False,
    )

    phase6 = al.PhaseImaging(
        phase_name="phase_6_initialize_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase5.result.instance.galaxies.lens.light,
                mass=phase5.result.instance.galaxies.lens.mass,
                shear=phase5.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase5.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_pixelization,
                regularization=pipeline_regularization,
            ),
        ),
        hyper_image_sky=phase5.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase5.result.hyper_combined.instance.hyper_background_noise,
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
                light=phase5.result.model.galaxies.lens.light,
                mass=phase5.result.model.galaxies.lens.mass,
                shear=phase5.result.model.galaxies.lens.shear,
                hyper_galaxy=phase6.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase6.result.instance.galaxies.source.pixelization,
                regularization=phase6.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase6.result.hyper_combined.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase6.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase6.result.hyper_combined.instance.hyper_background_noise,
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

    return al.PipelineDataset(
        name, phase1, phase2, phase3, phase4, phase5, phase6, phase7, hyper_mode=False
    )


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
