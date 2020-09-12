import autofit as af
import autolens as al


class TestSLaMPipelineSource:
    def test__shear(self):

        setup_mass = al.SetupMassTotal(no_shear=False)
        pipeline_source = al.SLaMPipelineSourceParametric(setup_mass=setup_mass)

        assert pipeline_source.shear is al.mp.ExternalShear

        setup_mass = al.SetupMassTotal(no_shear=True)
        pipeline_source = al.SLaMPipelineSourceParametric(setup_mass=setup_mass)

        assert pipeline_source.shear == None


class TestSLaMPipelineMass:
    def test__light_is_model_tag(self):

        pipeline_mass = al.SLaMPipelineMass(light_is_model=False)
        assert pipeline_mass.light_is_model_tag == ""
        pipeline_mass = al.SLaMPipelineMass(light_is_model=True)
        assert pipeline_mass.light_is_model_tag == "__light_is_model"

    def test__shear_from_previous_pipeline(self):

        setup_mass = al.SetupMassTotal(no_shear=True)
        pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

        assert pipeline_mass.shear_from_previous_pipeline == None

        setup_mass = al.SetupMassTotal(no_shear=False)
        pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

        assert isinstance(
            pipeline_mass.shear_from_previous_pipeline, af.AbstractPromise
        )
