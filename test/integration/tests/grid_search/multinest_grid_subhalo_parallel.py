import os

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'grid_search'
test_name = "multinest_grid_subhalo_parallel"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='no_lens_light_and_source_smooth', data_resolution='AO')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):

    class GridPhase(af.phase.as_grid_search(phase_imaging.LensSourcePlanePhase, parallel=True)):

        @property
        def grid_priors(self):
            return [self.variable.lens_galaxies.subhalo.mass.centre_0,
                    self.variable.lens_galaxies.subhalo.mass.centre_1]

        def pass_priors(self, results):

            ### Lens Subhalo, Adjust priors to physical masses (10^6 - 10^10) and concentrations (6-24)

            self.lens_galaxies.subhalo.mass.kappa_s = af.prior.UniformPrior(lower_limit=0.0005, upper_limit=0.2)
            self.lens_galaxies.subhalo.mass.scale_radius = af.prior.UniformPrior(lower_limit=0.001, upper_limit=1.0)
            self.lens_galaxies.subhalo.mass.centre_0 = af.prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.lens_galaxies.subhalo.mass.centre_1 = af.prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)


    phase1 = GridPhase(
        phase_name='phase_1', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal),
                           subhalo=gm.GalaxyModel(redshift=0.5, mass=mp.SphericalTruncatedNFWChallenge)),
        source_galaxies=dict(source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic)),
        optimizer_class=af.MultiNest, number_of_steps=2)

    phase1.optimizer.const_efficiency_mode = True

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
