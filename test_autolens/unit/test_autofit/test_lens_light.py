import autofit as af
import autolens as al

redshift_lens = 0.5,
redshift_source = 1.0,


def make_pipeline_with_lens_light():
    pipeline_name = "pipeline__with_lens_light__test"

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=al.lp.EllipticalSersic,
        disk=al.lp.EllipticalExponential,
    )

    lens.bulge.centre = lens.disk.centre

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        galaxies=dict(lens=lens),
        optimizer_class=af.MultiNest,
    )

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase1.result.instance.galaxies.lens.bulge,
                disk=phase1.result.instance.galaxies.lens.disk,
                mass=mass,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=al.lp.EllipticalSersic,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=al.lp.EllipticalSersic,
        disk=al.lp.EllipticalExponential,
        mass=phase2.result.instance.galaxies.lens.mass,
        shear=phase2.result.instance.galaxies.lens.shear,
        hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    lens.bulge.centre = lens.disk.centre

    phase3 = al.PhaseImaging(
        phase_name="phase_3",
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=phase2.result.instance.galaxies.source.sersic,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase4 = al.PhaseImaging(
        phase_name="phase_4",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase3.result.model.galaxies.lens.bulge,
                disk=phase3.result.model.galaxies.lens.disk,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=phase2.result.model.galaxies.source.sersic,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase5 = al.PhaseImaging(
        phase_name="phase_5",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=af.last.instance.galaxies.lens.bulge,
                disk=af.last.instance.galaxies.lens.disk,
                mass=af.last.instance.galaxies.lens.mass,
                shear=af.last.instance.galaxies.lens.shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase6 = al.PhaseImaging(
        phase_name="phase_6",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=af.last[-1].instance.galaxies.lens.bulge,
                disk=af.last[-1].instance.galaxies.lens.disk,
                mass=af.last[-1].model.galaxies.lens.mass,
                shear=af.last[-1].model.galaxies.lens.shear,
                hyper_galaxy=phase5.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase5.result.instance.galaxies.source.pixelization,
                regularization=phase5.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase5.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase5.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase5.result.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase7 = al.PhaseImaging(
        phase_name="phase_7",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase6.result.instance.galaxies.lens.bulge,
                disk=phase6.result.instance.galaxies.lens.disk,
                mass=phase6.result.instance.galaxies.lens.mass,
                shear=phase6.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase6.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
                hyper_galaxy=phase6.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase6.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase6.result.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.axis_ratio = phase2.result.model.galaxies.lens.mass.axis_ratio
    mass.phi = phase2.result.model.galaxies.lens.mass.phi
    mass.einstein_radius = phase2.result.model.galaxies.lens.mass.einstein_radius

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=phase6.result.instance.galaxies.lens.bulge,
        disk=phase6.result.instance.galaxies.lens.disk,
        mass=mass,
        shear=phase6.result.model.galaxies.lens.shear,
        hyper_galaxy=phase7.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    phase8 = al.PhaseImaging(
        phase_name="phase_8",
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase7.result.instance.galaxies.source.pixelization,
                regularization=phase7.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase7.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase7.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase7.result.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8)


def test__pipeline_with_lens_light__has_correct_free_parameters():
    pipeline = make_pipeline_with_lens_light()

    # 7 Bulge + 4 Disk (aligned centres)
    assert pipeline.phases[0].model.prior_count == 11

    # 5 Lens SIE + 12 Source Sersic
    assert pipeline.phases[1].model.prior_count == 12

    # 7 Bulge + 4 Disk (aligned centres)
    assert pipeline.phases[2].model.prior_count == 11

    # 7 Bulge + 4 Disk (aligned centres) + 5 Lens SIE + 7 Source Sersic
    assert pipeline.phases[3].model.prior_count == 23

    # 3 Source Inversion
    assert pipeline.phases[4].model.prior_count == 3

    # 5 Lens SIE
    assert pipeline.phases[5].model.prior_count == 3

    # 6 Source Inversion
    assert pipeline.phases[6].model.prior_count == 6

    # 5 SIE
    assert pipeline.phases[7].model.prior_count == 5
