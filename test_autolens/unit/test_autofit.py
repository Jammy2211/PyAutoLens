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

def make_pipeline_no_lens_light_hyper():

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

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_noise=True,
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

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_noise=True,
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

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_noise=True,
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

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_noise=True,
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

    phase5 = phase5.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_noise=True,
        inversion=True
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

    phase6 = phase6.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_noise=True,
        inversion=True
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4, phase5, phase6)

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


class TestAutofit:

    def test__pipeline_no_lens_light__has_correct_free_parameters(self):

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

    def test__pipeline_no_lens_light_hyper__has_correct_free_parameters(self):

        pipeline = make_pipeline_no_lens_light_hyper()

        # 5 Lens SIE + 12 Source Sersic
        assert pipeline.phases[0].model.prior_count == 12
        # 3 Hyper Galaxy + 1 Hyper BG Noise
        assert pipeline.phases[0].hyper_combined.model.prior_count == 4

        # 3 Source Inversion
        assert pipeline.phases[1].model.prior_count == 3
        # 3 Hyper Galaxy + 1 Hyper BG Noise
        assert pipeline.phases[1].hyper_combined.model.prior_count == 4

        # 5 Lens SIE
        assert pipeline.phases[2].model.prior_count == 5
        # 3 Hyper Galaxy + 1 Hyper BG Noise
        assert pipeline.phases[2].hyper_combined.model.prior_count == 4

        # 6 Source Inversion
        assert pipeline.phases[3].model.prior_count == 6
        # 3 Hyper Galaxy + 1 Hyper BG Noise
        assert pipeline.phases[3].hyper_combined.model.prior_count == 4

        # 5 Lens SIE
        assert pipeline.phases[4].model.prior_count == 5
        # 3 Hyper Galaxy + 1 Hyper BG Noise + 6 Inversion
        assert pipeline.phases[4].hyper_combined.model.prior_count == 10

        # 6 Lens SPLE
        assert pipeline.phases[5].model.prior_count == 6
        # 3 Hyper Galaxy + 1 Hyper BG Noise + 6 Inversion
        assert pipeline.phases[5].hyper_combined.model.prior_count == 10

    def test__pipeline_with_lens_light__has_correct_free_parameters(self):

        pipeline = make_pipeline_no_lens_light()

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