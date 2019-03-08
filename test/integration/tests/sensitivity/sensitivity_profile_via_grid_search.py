import os

from autofit import conf
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'sensitivity'
test_name = "sensitivity_profile_via_grid_search"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_resolution='Euclid', data_name='no_lens_light_and_source_smooth')
    pipeline = make_pipeline(test_name=test_name)
    result = pipeline.run(data=ccd_data)

    print(dir(result))


def make_pipeline(test_name):

    class SensitivePhase(ph.SensitivityPhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.mass.centre_0 = 0.0
            self.lens_galaxies.lens.mass.centre_1 = 0.0
            self.lens_galaxies.lens.mass.einstein_radius = 1.6

            self.source_galaxies.source.light.centre_0 = 0.0
            self.source_galaxies.source.light.centre_1 = 0.0
            self.source_galaxies.source.light.intensity = 1.0
            self.source_galaxies.source.light.effective_radius = 0.5
            self.source_galaxies.source.light.sersic_index = 1.0

            self.sensitive_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.sensitive_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.sensitive_galaxies.subhalo.mass.kappa_s = 0.1
            self.sensitive_galaxies.subhalo.mass.scale_radius = 5.0

    phase1 = SensitivePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.SphericalIsothermal)),
                            source_galaxies=dict(source=gm.GalaxyModel(light=lp.SphericalSersic)),
                            sensitive_galaxies=dict(subhalo=gm.GalaxyModel(mass=mp.SphericalNFW)),
                            optimizer_class=nl.GridSearch, phase_name="{}/phase1".format(test_name))

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
