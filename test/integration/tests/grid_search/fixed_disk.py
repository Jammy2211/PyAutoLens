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
test_name = "fixed_disk"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path + 'output/' + test_type
config_path = path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():
    bulge = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0, intensity=1.0, effective_radius=1.3,
                                sersic_index=3.0)

    disk = lp.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=0.7, phi=0.0, intensity=2.0, effective_radius=2.0)

    lens_galaxy = galaxy.Galaxy(bulge=bulge, disk=disk)

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

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
