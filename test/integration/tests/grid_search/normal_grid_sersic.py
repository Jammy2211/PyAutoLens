import os

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'grid_search'
test_name = "normal_grid_sersic"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


def pipeline():
                                
    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='lens_only_dev_vaucouleurs', data_resolution='Euclid')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):
    
    class QuickPhase(phase_imaging.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.light.centre_0 = af.prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.light.centre_1 = af.prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.light.axis_ratio = af.prior.UniformPrior(lower_limit=0.79, upper_limit=0.81)
            self.lens_galaxies.lens.light.phi = af.prior.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
            self.lens_galaxies.lens.light.intensity = af.prior.UniformPrior(lower_limit=0.99, upper_limit=1.01)
            self.lens_galaxies.lens.light.effective_radius = af.prior.UniformPrior(lower_limit=1.25, upper_limit=1.35)
            self.lens_galaxies.lens.light.sersic_index = af.prior.UniformPrior(lower_limit=3.95, upper_limit=4.05)

    phase1 = QuickPhase(
        phase_name='phase_1', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic)),
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class GridPhase(phase_imaging.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.light.centre_0 = 0.0
            self.lens_galaxies.lens.light.centre_1 = 0.0
            self.lens_galaxies.lens.light.axis_ratio = results.from_phase('phase_1').constant.lens.light.axis_ratio
            self.lens_galaxies.lens.light.phi = results.from_phase('phase_1').constant.lens.light.phi
            self.lens_galaxies.lens.light.intensity = results.from_phase('phase_1').constant.lens.light.intensity

            self.lens_galaxies.lens.light.effective_radius = af.prior.UniformPrior(lower_limit=0.0, upper_limit=4.0)
            self.lens_galaxies.lens.light.sersic_index = af.prior.UniformPrior(lower_limit=1.0, upper_limit=8.0)

    phase2 = GridPhase(
        phase_name='phase_2', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic)),
        optimizer_class=af.GridSearch)

    phase2.optimizer.const_efficiency_mode = True

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
