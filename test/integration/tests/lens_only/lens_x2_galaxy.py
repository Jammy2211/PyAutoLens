import os

from autofit import conf
from autofit.core import non_linear as nl
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import tools

test_type = 'lens_only'
test_name = "lens_x2_galaxy"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    sersic_0 = lp.EllipticalSersic(centre=(-1.0, -1.0), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                   effective_radius=1.3, sersic_index=3.0)

    sersic_1 = lp.EllipticalSersic(centre=(1.0, 1.0), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                   effective_radius=1.3, sersic_index=3.0)

    lens_galaxy_0 = galaxy.Galaxy(light_profile=sersic_0)
    lens_galaxy_1 = galaxy.Galaxy(light_profile=sersic_1)

    tools.reset_paths(test_name=test_name, output_path=output_path)
    tools.simulate_integration_image(test_name=test_name, pixel_scale=0.1, lens_galaxies=[lens_galaxy_0, lens_galaxy_1],
                                     source_galaxies=[], target_signal_to_noise=30.0)
    image = tools.load_image(test_name=test_name, pixel_scale=0.1)

    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(image=image)

def make_pipeline(test_name):
    
    class LensPlanex2GalPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):
            self.lens_galaxies[0].sersic.centre_0 = -1.0
            self.lens_galaxies[0].sersic.centre_1 = -1.0
            self.lens_galaxies[1].sersic.centre_0 = 1.0
            self.lens_galaxies[1].sersic.centre_1 = 1.0

    def modify_mask_function(img):
        return msk.Mask.circular(shape=img.shape, pixel_scale=img.pixel_scale, radius_arcsec=5.)

    phase1 = LensPlanex2GalPhase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                 mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                 phase_name="{}/phase1".format(test_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
