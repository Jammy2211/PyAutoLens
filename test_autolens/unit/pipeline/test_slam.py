import autofit as af
import autolens as al


class TestSLaMPipelineSource:
    def test__shear(self):

        setup_mass = al.SetupMassTotal(no_shear=False)
        pipeline_source = al.SLaMPipelineSourceParametric(setup_mass=setup_mass)

        assert isinstance(pipeline_source.setup_mass.shear_prior_model, af.PriorModel)

        setup_mass = al.SetupMassTotal(no_shear=True)
        pipeline_source = al.SLaMPipelineSourceParametric(setup_mass=setup_mass)

        assert pipeline_source.setup_mass.shear_prior_model == None


class TestSLaMPipelineMass:
    def test__light_is_model_tag(self):

        pipeline_mass = al.SLaMPipelineMass(light_is_model=False)
        assert pipeline_mass.light_is_model_tag == ""
        pipeline_mass = al.SLaMPipelineMass(light_is_model=True)
        assert pipeline_mass.light_is_model_tag == "__light_is_model"

    def test__shear_from_previous_pipeline(self):

        setup_mass = al.SetupMassTotal(no_shear=True)
        pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

        assert pipeline_mass.shear_from_previous_pipeline() == None

        setup_mass = al.SetupMassTotal(no_shear=False)
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
            f"source[parametric__bulge_sersic__disk_exp]"
        )

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

        slam = al.SLaM(
            pipeline_source_parametric=pipeline_source_parametric,
            pipeline_light_parametric=al.SLaMPipelineLightParametric(),
            pipeline_mass=al.SLaMPipelineMass(),
        )

        assert (
            slam.source_parametric_tag == f"source__"
            f"light[parametric__bulge_exp_sph]__"
            f"mass[total__power_law__with_shear__mass_centre_(0.00,0.00)]__"
            f"source[parametric__bulge_dev_sph__disk_exp]"
        )
