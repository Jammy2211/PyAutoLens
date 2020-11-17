import autofit as af
import autolens as al


class TestSLaMPipelineSource:
    def test__shear(self):

        setup_mass = al.SetupMassTotal(with_shear=True)
        pipeline_source = al.SLaMPipelineSourceParametric(setup_mass=setup_mass)

        assert isinstance(pipeline_source.setup_mass.shear_prior_model, af.PriorModel)

        setup_mass = al.SetupMassTotal(with_shear=False)
        pipeline_source = al.SLaMPipelineSourceParametric(setup_mass=setup_mass)

        assert pipeline_source.setup_mass.shear_prior_model == None


class TestSLaMPipelineMass:
    def test__light_is_model_tag(self):

        pipeline_mass = al.SLaMPipelineMass(light_is_model=False)
        assert pipeline_mass.light_is_model_tag == "__light_is_instance"
        pipeline_mass = al.SLaMPipelineMass(light_is_model=True)
        assert pipeline_mass.light_is_model_tag == "__light_is_model"

    def test__shear_from_previous_pipeline(self):

        setup_mass = al.SetupMassTotal(with_shear=False)
        pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

        assert pipeline_mass.shear_from_previous_pipeline() == None

        setup_mass = al.SetupMassTotal(with_shear=True)
        pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

        assert isinstance(
            pipeline_mass.shear_from_previous_pipeline(), af.AbstractPromise
        )


class TestSLaM:
    def test__source_parametric_tag(self):

        slam = al.SLaM(
            pipeline_source_parametric=al.SLaMPipelineSourceParametric(),
            pipeline_mass=al.SLaMPipelineMass(),
        )

        assert (
            slam.source_parametric_tag == f"source__"
            f"mass[total__sie__with_shear]__"
            f"source[parametric__bulge_sersic]"
        )
        assert slam.source_parametric_tag == slam.source_tag

        pipeline_source_parametric = al.SLaMPipelineSourceParametric(
            setup_light=al.SetupLightParametric(
                bulge_prior_model=al.lp.SphericalExponential,
                disk_prior_model=None,
                align_bulge_disk_centre=False,
            ),
            setup_mass=al.SetupMassTotal(
                mass_prior_model=al.mp.EllipticalPowerLaw, mass_centre=(0.0, 0.0)
            ),
            setup_source=al.SetupSourceParametric(
                bulge_prior_model=al.lp.SphericalDevVaucouleurs,
                align_bulge_disk_centre=False,
            ),
        )

        slam = al.SLaM(
            pipeline_source_parametric=pipeline_source_parametric,
            pipeline_light_parametric=al.SLaMPipelineLightParametric(),
            pipeline_mass=al.SLaMPipelineMass(),
        )

        assert (
            slam.source_parametric_tag == f"source__"
            f"light[parametric__bulge_exp_sph]__"
            f"mass[total__power_law__with_shear__centre_(0.00,0.00)]__"
            f"source[parametric__bulge_dev_sph]"
        )
        assert slam.source_parametric_tag == slam.source_tag

    def test__source_inversion_tag(self):

        slam = al.SLaM(
            pipeline_source_parametric=al.SLaMPipelineSourceParametric(),
            pipeline_source_inversion=al.SLaMPipelineSourceInversion(),
            pipeline_mass=al.SLaMPipelineMass(),
        )

        assert (
            slam.source_inversion_tag == f"source__"
            f"mass[total__sie__with_shear]__"
            f"source[inversion__pix_rect__reg_const]"
        )
        assert slam.source_inversion_tag == slam.source_tag

        pipeline_source_parametric = al.SLaMPipelineSourceParametric(
            setup_light=al.SetupLightParametric(
                bulge_prior_model=al.lp.SphericalExponential, disk_prior_model=None
            ),
            setup_mass=al.SetupMassTotal(
                mass_prior_model=al.mp.EllipticalPowerLaw, mass_centre=(0.0, 0.0)
            ),
            setup_source=al.SetupSourceParametric(
                bulge_prior_model=al.lp.SphericalDevVaucouleurs
            ),
        )

        pipeline_source_inversion = al.SLaMPipelineSourceInversion(
            setup_source=al.SetupSourceInversion(
                pixelization_prior_model=al.pix.VoronoiMagnification,
                regularization_prior_model=al.reg.AdaptiveBrightness,
            )
        )

        slam = al.SLaM(
            pipeline_source_parametric=pipeline_source_parametric,
            pipeline_source_inversion=pipeline_source_inversion,
            pipeline_light_parametric=al.SLaMPipelineLightParametric(),
            pipeline_mass=al.SLaMPipelineMass(),
        )

        assert (
            slam.source_inversion_tag == f"source__"
            f"light[parametric__bulge_exp_sph]__"
            f"mass[total__power_law__with_shear__centre_(0.00,0.00)]__"
            f"source[inversion__pix_voro_mag__reg_adapt_bright]"
        )
        assert slam.source_inversion_tag == slam.source_tag

    def test__light_parametric_tag(self):

        slam = al.SLaM(
            pipeline_source_parametric=al.SLaMPipelineSourceParametric(),
            pipeline_light_parametric=al.SLaMPipelineLightParametric(),
            pipeline_mass=al.SLaMPipelineMass(),
        )

        assert (
            slam.light_parametric_tag == f"light__"
            f"light[parametric__bulge_sersic__disk_exp__align_bulge_disk_centre]__"
            f"mass[total__sie__with_shear]__"
            f"source[parametric__bulge_sersic]"
        )

        pipeline_source_parametric = al.SLaMPipelineSourceParametric(
            setup_light=al.SetupLightParametric(
                bulge_prior_model=al.lp.SphericalExponential,
                disk_prior_model=None,
                align_bulge_disk_centre=False,
            ),
            setup_mass=al.SetupMassTotal(
                mass_prior_model=al.mp.EllipticalPowerLaw, mass_centre=(0.0, 0.0)
            ),
            setup_source=al.SetupSourceParametric(
                bulge_prior_model=al.lp.SphericalDevVaucouleurs
            ),
        )

        pipeline_source_inversion = al.SLaMPipelineSourceInversion(
            setup_source=al.SetupSourceInversion(
                pixelization_prior_model=al.pix.VoronoiMagnification,
                regularization_prior_model=al.reg.AdaptiveBrightness,
            )
        )

        pipeline_light_parametric = al.SLaMPipelineLightParametric(
            setup_light=al.SetupLightParametric(
                bulge_prior_model=al.lp.SphericalDevVaucouleurs,
                disk_prior_model=al.lp.SphericalExponential,
                align_bulge_disk_centre=False,
                light_centre=(0.0, 0.0),
            )
        )

        slam = al.SLaM(
            pipeline_source_parametric=pipeline_source_parametric,
            pipeline_source_inversion=pipeline_source_inversion,
            pipeline_light_parametric=pipeline_light_parametric,
            pipeline_mass=al.SLaMPipelineMass(),
        )

        assert (
            slam.light_parametric_tag == f"light__"
            f"light[parametric__bulge_dev_sph__disk_exp_sph__centre_(0.00,0.00)]__"
            f"mass[total__power_law__with_shear__centre_(0.00,0.00)]__"
            f"source[inversion__pix_voro_mag__reg_adapt_bright]"
        )

    def test__mass_tag(self):

        slam = al.SLaM(
            pipeline_source_parametric=al.SLaMPipelineSourceParametric(),
            pipeline_light_parametric=al.SLaMPipelineLightParametric(),
            pipeline_mass=al.SLaMPipelineMass(),
        )

        assert (
            slam.mass_tag == f"mass__"
            f"light[parametric__bulge_sersic__disk_exp__align_bulge_disk_centre]__"
            f"mass[total__power_law__with_shear]__"
            f"source[parametric__bulge_sersic]"
        )

        pipeline_source_parametric = al.SLaMPipelineSourceParametric(
            setup_light=al.SetupLightParametric(
                bulge_prior_model=al.lp.SphericalExponential,
                disk_prior_model=None,
                align_bulge_disk_centre=False,
            ),
            setup_mass=al.SetupMassTotal(
                mass_prior_model=al.mp.EllipticalPowerLaw, mass_centre=(0.0, 0.0)
            ),
            setup_source=al.SetupSourceParametric(
                bulge_prior_model=al.lp.SphericalDevVaucouleurs
            ),
        )

        pipeline_source_inversion = al.SLaMPipelineSourceInversion(
            setup_source=al.SetupSourceInversion(
                pixelization_prior_model=al.pix.VoronoiMagnification,
                regularization_prior_model=al.reg.AdaptiveBrightness,
            )
        )

        pipeline_light_parametric = al.SLaMPipelineLightParametric(
            setup_light=al.SetupLightParametric(
                bulge_prior_model=al.lp.SphericalDevVaucouleurs,
                disk_prior_model=al.lp.SphericalExponential,
                align_bulge_disk_centre=False,
                light_centre=(0.0, 0.0),
            )
        )

        pipeline_mass = al.SLaMPipelineMass(
            setup_mass=al.SetupMassLightDark(
                bulge_prior_model=al.lmp.EllipticalSersicRadialGradient
            )
        )

        slam = al.SLaM(
            pipeline_source_parametric=pipeline_source_parametric,
            pipeline_source_inversion=pipeline_source_inversion,
            pipeline_light_parametric=pipeline_light_parametric,
            pipeline_mass=pipeline_mass,
        )

        assert (
            slam.mass_tag == f"mass__"
            f"light[parametric__bulge_dev_sph__disk_exp_sph__centre_(0.00,0.00)]__"
            f"mass[light_dark__bulge_sersic_grad__disk_exp__mlr_free__dark_nfw_sph_ludlow__with_shear]__"
            f"source[inversion__pix_voro_mag__reg_adapt_bright]"
        )
