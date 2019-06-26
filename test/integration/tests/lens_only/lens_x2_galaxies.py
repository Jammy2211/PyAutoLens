import os

import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_only'
test_name = "lens_x2_galaxies"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='lens_only_x2_galaxies', data_resolution='LSST')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)

def make_pipeline(test_name):
    
    class LensPlanex2GalPhase(phase_imaging.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens_0.light.centre_0 = -1.0
            self.lens_galaxies.lens_0.light.centre_1 = -1.0
            self.lens_galaxies.lens_1.light.centre_0 = 1.0
            self.lens_galaxies.lens_1.light.centre_1 = 1.0

    def mask_function(image):
        return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=5.)

    phase1 = LensPlanex2GalPhase(phase_name='phase_1', phase_folders=[test_type, test_name],
                                 lens_galaxies=dict(lens_0=gm.GalaxyModel(light=lp.EllipticalSersic),
                                                    lens_1=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                 mask_function=mask_function, optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
