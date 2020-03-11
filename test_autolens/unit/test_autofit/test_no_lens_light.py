import autofit as af
import autolens as al
from autofit.optimize.non_linear.mock_nlo import MockNLO

redshift_lens = 0.5,
redshift_source = 1.0,


def make_pipeline_no_lens_light():
    pipeline_name = "pipeline__no_lens_light__test"

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens, mass=al.mp.EllipticalIsothermal,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        optimizer_class=MockNLO,
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=af.last.instance.galaxies.lens.mass,
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
        optimizer_class=MockNLO,
    )

    phase3 = al.PhaseImaging(
        phase_name="phase_3",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=af.last.model.galaxies.lens.mass,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        optimizer_class=MockNLO,
    )

    phase4 = al.PhaseImaging(
        phase_name="phase_4",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase2.result.instance.galaxies.lens.mass,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=MockNLO,
    )

    phase5 = al.PhaseImaging(
        phase_name="phase_5",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase2.result.model.galaxies.lens.mass,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase4.result.instance.galaxies.source.pixelization,
                regularization=phase4.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase4.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase4.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase4.result.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=MockNLO,
    )

    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    mass.centre = af.last.model.galaxies.lens.mass.centre
    mass.axis_ratio = af.last.model.galaxies.lens.mass.axis_ratio
    mass.phi = af.last.model.galaxies.lens.mass.phi
    mass.einstein_radius = af.last.model_absolute(
        a=0.3
    ).galaxies.lens.mass.einstein_radius

    source = al.GalaxyModel(
        redshift=af.last.instance.galaxies.source.redshift,
        pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
        regularization=af.last.hyper_combined.instance.galaxies.source.regularization,
    )

    phase6 = al.PhaseImaging(
        phase_name="phase_6",
        galaxies=dict(
            lens=al.GalaxyModel(redshift=redshift_lens, mass=mass),
            source=source,
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        optimizer_class=MockNLO,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4, phase5, phase6)


def test__pipeline_no_lens_light__has_correct_free_parameters():
    pipeline = make_pipeline_no_lens_light()

    # 5 Lens SIE + 12 Source Sersic
    assert pipeline.phases[0].model.prior_count == 12

    # 3 Source Inversion
    assert pipeline.phases[1].model.prior_count == 3

    # 5 Lens SIE
    assert pipeline.phases[2].model.prior_count == 5

    # 6 Source Inversion
    assert pipeline.phases[3].model.prior_count == 6

    # 5 Lens SIE
    assert pipeline.phases[4].model.prior_count == 5

    # 6 Lens SPLE
    assert pipeline.phases[5].model.prior_count == 6
