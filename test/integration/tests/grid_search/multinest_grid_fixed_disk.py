import os

from autofit import conf
from autofit.mapper import prior
from autofit.optimize import non_linear as nl
from autofit.tools import phase as autofit_ph
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'grid_search'
test_name = "multinest_grid_fixed_disk"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path + 'output/' + test_type
config_path = path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_resolution='Euclid', data_name='lens_only_dev_vaucouleurs')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):
    class QuickPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.bulge.centre_0 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.bulge.centre_1 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.bulge.axis_ratio = prior.UniformPrior(lower_limit=0.79, upper_limit=0.81)
            self.lens_galaxies.lens.bulge.phi = prior.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
            self.lens_galaxies.lens.bulge.intensity = prior.UniformPrior(lower_limit=0.99, upper_limit=1.01)
            self.lens_galaxies.lens.bulge.effective_radius = prior.UniformPrior(lower_limit=1.25, upper_limit=1.35)
            self.lens_galaxies.lens.bulge.sersic_index = prior.UniformPrior(lower_limit=3.95, upper_limit=4.05)

            self.lens_galaxies.lens.disk.centre_0 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.disk.centre_1 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.disk.axis_ratio = prior.UniformPrior(lower_limit=0.69, upper_limit=0.71)
            self.lens_galaxies.lens.disk.phi = prior.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
            self.lens_galaxies.lens.disk.intensity = prior.UniformPrior(lower_limit=1.99, upper_limit=2.01)
            self.lens_galaxies.lens.disk.effective_radius = prior.UniformPrior(lower_limit=1.95, upper_limit=2.05)

    phase1 = QuickPhase(lens_galaxies=dict(lens=gm.GalaxyModel(bulge=lp.EllipticalSersic,
                                                               disk=lp.EllipticalExponential)),
                        optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(test_name))

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class GridPhase(autofit_ph.as_grid_search(ph.LensPlanePhase)):

        @property
        def grid_priors(self):
            return [self.variable.lens.bulge.sersic_index]

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.disk = previous_results[0].constant.lens.disk

            self.lens_galaxies.lens.bulge.centre_0 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.bulge.centre_1 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.bulge.axis_ratio = prior.UniformPrior(lower_limit=0.79, upper_limit=0.81)
            self.lens_galaxies.lens.bulge.phi = prior.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
            self.lens_galaxies.lens.bulge.intensity = prior.UniformPrior(lower_limit=0.99, upper_limit=1.01)
            self.lens_galaxies.lens.bulge.effective_radius = prior.UniformPrior(lower_limit=1.25, upper_limit=1.35)

    phase2 = GridPhase(lens_galaxies=dict(lens=gm.GalaxyModel(bulge=lp.EllipticalSersic,
                                                              disk=lp.EllipticalExponential)),
                       number_of_steps=2, optimizer_class=nl.MultiNest,
                       phase_name=test_name + '/phase2')

    phase2.optimizer.const_efficiency_mode = True

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
