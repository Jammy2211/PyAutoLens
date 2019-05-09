import os

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_only'
test_name = "lens_x2_galaxies_separate"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='lens_only_x2_galaxies', data_resolution='LSST')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data, assert_optimizer_pickle_matches=False)

def make_pipeline(test_name):
    
    def modify_mask_function(image):
        return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=5.)

    class LensPlaneGalaxy0Phase(ph.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens_0.sersic.centre_0 = -1.0
            self.lens_galaxies.lens_0.sersic.centre_1 = -1.0

    phase1 = LensPlaneGalaxy0Phase(phase_name='phase_1', phase_folders=[test_type, test_name],
                                   lens_galaxies=dict(lens_0=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class LensPlaneGalaxy1Phase(ph.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens_0 = results.from_phase('phase_1').\
                constant.lens_galaxies.lens_0

            self.lens_galaxies.lens_1.sersic.centre_0 = 1.0
            self.lens_galaxies.lens_1.sersic.centre_1 = 1.0

    phase2 = LensPlaneGalaxy1Phase(phase_name='phase_2', phase_folders=[test_type, test_name],
                                   lens_galaxies=dict(lens_0=gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                      lens_1=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    class LensPlaneBothGalaxyPhase(ph.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens_0 = results.from_phase('phase_1').\
                variable.lens_galaxies.lens_0

            self.lens_galaxies.lens_1 = results.from_phase('phase_2').\
                variable.lens_galaxies.lens_0

            self.lens_galaxies.lens_0.sersic.centre_0 = -1.0
            self.lens_galaxies.lens_0.sersic.centre_1 = -1.0
            self.lens_galaxies.lens_1.sersic.centre_0 = 1.0
            self.lens_galaxies.lens_1.sersic.centre_1 = 1.0

    phase3 = LensPlaneBothGalaxyPhase(phase_name='phase_3', phase_folders=[test_type, test_name],
                                      lens_galaxies=dict(lens_0=gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                         lens_1=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                                      mask_function=modify_mask_function, optimizer_class=nl.MultiNest)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase2, phase3)


if __name__ == "__main__":
    pipeline()
