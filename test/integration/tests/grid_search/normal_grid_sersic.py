import os

from autofit import conf
from autofit.mapper import prior
from autofit.optimize import non_linear as nl
from autofit.tools import phase as autofit_ph
from autolens.data import ccd
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from test.integration import tools

test_type = 'grid_search'
test_name = "normal_grid_sersic"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path + 'output/' + test_type
config_path = path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():
    
    sersic = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, intensity=1.0, effective_radius=1.3,
                                sersic_index=3.0)

    lens_galaxy = galaxy.Galaxy(light=sersic)
                                
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
    
    class QuickPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):
            self.lens_galaxies.lens.light.centre_0 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.light.centre_1 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.light.axis_ratio = prior.UniformPrior(lower_limit=0.79, upper_limit=0.81)
            self.lens_galaxies.lens.light.phi = prior.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
            self.lens_galaxies.lens.light.intensity = prior.UniformPrior(lower_limit=0.99, upper_limit=1.01)
            self.lens_galaxies.lens.light.effective_radius = prior.UniformPrior(lower_limit=1.25, upper_limit=1.35)
            self.lens_galaxies.lens.light.sersic_index = prior.UniformPrior(lower_limit=3.95, upper_limit=4.05)

    phase1 = QuickPhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                        optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(test_name))

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class GridPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.light = previous_results[0].constant.lens.light

            self.lens_galaxies.lens.light.effective_radius = prior.UniformPrior(lower_limit=0.0, upper_limit=4.0)
            self.lens_galaxies.lens.light.sersic_index = prior.UniformPrior(lower_limit=1.0, upper_limit=8.0)

    phase2 = GridPhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                       optimizer_class=nl.GridSearch, phase_name=test_type + '/phase2')

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
