import os
import shutil

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.data import ccd
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.integration import tools

test_type = 'lens_and_source'
test_name = "lens_x1_source_x2"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    lens_mass = mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=80.0, einstein_radius=1.6)
    source_light_0 = lp.EllipticalSersic(centre=(-0.6, 0.5), axis_ratio=0.6, phi=60.0, intensity=1.0,
                                         effective_radius=0.5, sersic_index=1.0)
    source_light_1 = lp.EllipticalSersic(centre=(0.2, 0.3), axis_ratio=0.6, phi=90.0, intensity=1.0,
                                         effective_radius=0.5, sersic_index=1.0)

    lens_galaxy = galaxy.Galaxy(sie=lens_mass)
    source_galaxy_0 = galaxy.Galaxy(sersic=source_light_0)
    source_galaxy_1 = galaxy.Galaxy(sersic=source_light_1)

    tools.reset_paths(test_name=test_name, output_path=output_path)
    tools.simulate_integration_image(test_name=test_name, pixel_scale=0.1, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[source_galaxy_0, source_galaxy_1], target_signal_to_noise=30.0)

    ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/' + test_name + '/image.fits',
                                        psf_path=path + '/data/' + test_name + '/psf.fits',
                                        noise_map_path=path + '/data/' + test_name + '/noise_map.fits',
                                        pixel_scale=0.1)

    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)

def make_pipeline(test_name):

    phase1 = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source_0=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(test_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.7

    class AddSourceGalaxyPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies_lens = previous_results[0].variable.lens
            self.source_galaxies_source_0 = previous_results[0].variable.source_0

    phase2 = AddSourceGalaxyPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                  source_galaxies=dict(source_0=gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                       source_1=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                                  optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(test_name))

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.7

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
