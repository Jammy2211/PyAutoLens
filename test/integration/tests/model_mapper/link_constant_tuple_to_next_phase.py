import os

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'model_mapper'
test_name = "link_constant_tuple_to_next_phase"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='lens_only_dev_vaucouleurs', data_resolution='LSST')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):

    class MMPhase(phase_imaging.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.light.centre_0 = 1.0
            self.lens_galaxies.lens.light.centre_1 = 2.0

    phase1 = MMPhase(
        phase_name='phase_1', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic)),
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    class MMPhase2(phase_imaging.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.light.centre_0 = results.from_phase('phase_1').\
                variable.lens_galaxies.lens.light.centre_0

            self.lens_galaxies.lens.light.centre_1 = results.from_phase('phase_1').\
                variable.lens_galaxies.lens.light.centre_1

    phase2 = MMPhase2(
        phase_name='phase_2', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic)),
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
