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
    def test__fix_lens_light_tag(self):

        pipeline_mass = al.SLaMPipelineMass(fix_lens_light=False)
        assert pipeline_mass.fix_lens_light_tag == ""
        pipeline_mass = al.SLaMPipelineMass(fix_lens_light=True)
        assert pipeline_mass.fix_lens_light_tag == "__fix_lens_light"

    def test__shear_from_previous_pipeline(self):

        setup_mass = al.SetupMassTotal(no_shear=True)
        pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

        assert pipeline_mass.shear_from_previous_pipeline == None

        setup_mass = al.SetupMassTotal(no_shear=False)
        pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

        assert isinstance(
            pipeline_mass.shear_from_previous_pipeline, af.AbstractPromise
        )
