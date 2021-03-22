import autofit as af
import autolens as al


class TestSLaMPipelineSource:
    def test__shear(self):

        setup_mass = al.SetupMassTotal(with_shear=True)
        pipeline_source = al.SLaMPipelineSourceParametric(setup_mass=setup_mass)

        assert isinstance(pipeline_source.setup_mass.shear_prior_model, af.Model)

        setup_mass = al.SetupMassTotal(with_shear=False)
        pipeline_source = al.SLaMPipelineSourceParametric(setup_mass=setup_mass)

        assert pipeline_source.setup_mass.shear_prior_model == None


class TestSLaMPipelineMass:
    def test__shear_from_previous_pipeline(self):

        setup_mass = al.SetupMassTotal(with_shear=False)
        pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

        assert pipeline_mass.shear_from_result(result=None) == None
