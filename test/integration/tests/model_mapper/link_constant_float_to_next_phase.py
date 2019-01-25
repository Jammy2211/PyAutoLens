import os
import shutil

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.data import ccd
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import tools

test_type = 'model_mapper'
test_name = "link_constant_float_to_next_phase"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    sersic = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=1.0, effective_radius=1.3,
                                 sersic_index=3.0)

    lens_galaxy = galaxy.Galaxy(light_profile=sersic)

    tools.reset_paths(test_name=test_name, output_path=output_path)
    tools.simulate_integration_image(test_name=test_name, pixel_scale=0.1, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[], target_signal_to_noise=30.0)

    ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/' + test_name + '/image.fits',
                                        psf_path=path + '/data/' + test_name + '/psf.fits',
                                        noise_map_path=path + '/data/' + test_name + '/noise_map.fits',
                                        pixel_scale=0.1)

    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):

    class MMPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):
            
            self.lens_galaxies.lens.light.axis_ratio = 0.2
            self.lens_galaxies.lens.light.phi = 90.0

    phase1 = MMPhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                     optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(test_name))

    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    class MMPhase2(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens = previous_results[0].variable.lens

    phase2 = MMPhase2(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                      optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(test_name))

    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
