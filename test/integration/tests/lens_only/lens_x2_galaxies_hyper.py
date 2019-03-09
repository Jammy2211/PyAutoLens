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
test_name = "lens_x2_galaxies_hyper"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def test_pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_resolution='LSST', data_name='lens_only_x2_galaxies')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)

def make_pipeline(test_name):
    def modify_mask_function(img):
        return msk.Mask.circular(shape=img.shape, pixel_scale=img.pixel_scale, radius_arcsec=5.)

    class LensPlaneGalaxy0Phase(ph.LensPlanePhase):
        
        def pass_priors(self, previous_results):
            
            self.lens_galaxies.lens_0.light.centre_0 = -1.0
            self.lens_galaxies.lens_0.light.centre_1 = -1.0

    phase1 = LensPlaneGalaxy0Phase(lens_galaxies=dict(lens_0=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                   phase_name="{}/phase1".format(test_name))

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class LensPlaneGalaxy1Phase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            
            self.lens_galaxies.lens_0 = previous_results[0].constant.lens_0
            self.lens_galaxies.lens_1.light.centre_0 = 1.0
            self.lens_galaxies.lens_1.light.centre_1 = 1.0

    phase2 = LensPlaneGalaxy1Phase(lens_galaxies=dict(lens_0=gm.GalaxyModel(light=lp.EllipticalSersic),
                                                      lens_1=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                   phase_name="{}/phase2".format(test_name))

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    phase2h = ph.LensLightHyperOnlyPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase2h".format(test_name),
                                         mask_function=modify_mask_function)

    class LensPlaneBothGalaxyPhase(ph.LensPlaneHyperPhase):
        def pass_priors(self, previous_results):
            
            self.lens_galaxies.lens_0 = previous_results[0].variable.lens_0
            self.lens_galaxies.lens_1 = previous_results[1].variable.lens_0
            self.lens_galaxies.lens_0.hyper_galaxy = previous_results[-1].hyper.constant.lens_0.hyper_galaxy
            self.lens_galaxies.lens_1.hyper_galaxy = previous_results[-1].hyper.constant.lens_1.hyper_galaxy
            self.lens_galaxies.lens_0.light.centre_0 = -1.0
            self.lens_galaxies.lens_0.light.centre_1 = -1.0
            self.lens_galaxies.lens_1.light.centre_0 = 1.0
            self.lens_galaxies.lens_1.light.centre_1 = 1.0

    phase3 = LensPlaneBothGalaxyPhase(lens_galaxies=dict(lens_0=gm.GalaxyModel(light=lp.EllipticalSersic),
                                                         lens_1=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                      mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                      phase_name="{}/phase3".format(test_name))

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase2, phase2h, phase3)


if __name__ == "__main__":
    test_pipeline()
