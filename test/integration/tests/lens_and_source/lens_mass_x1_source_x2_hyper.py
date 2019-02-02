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
test_name = "lens_mass_x1_source_x2_hyper"

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

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.7

    phase1h = ph.LensMassAndSourceProfileHyperOnlyPhase(optimizer_class=nl.MultiNest,
                                                        phase_name="{}/phase1h".format(test_name))

    class AddSourceGalaxyPhase(ph.LensSourcePlaneHyperPhase):
        
        def pass_priors(self, previous_results):
            phase1_results = previous_results[-1]
            phase1h_results = previous_results[-1].hyper
            self.lens_galaxies.lens = phase1_results.variable.lens
            self.source_galaxies.source_0 = phase1_results.variable.source_0
            self.source_galaxies.source_0.hyper_galaxy = phase1h_results.constant.source_0.hyper_galaxy

    phase2 = AddSourceGalaxyPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                  source_galaxies=dict(source_0=gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                       source_1=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                                  optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(test_name))

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.7

    phase2h = ph.LensMassAndSourceProfileHyperOnlyPhase(optimizer_class=nl.MultiNest,
                                                        phase_name="{}/phase1h".format(test_name))

    class BothSourceGalaxiesPhase(ph.LensSourcePlaneHyperPhase):

        def pass_priors(self, previous_results):

            phase2_results = previous_results[1]
            phase2h_results = previous_results[1].hyper
            self.lens_galaxies.lens = phase2_results.variable.lens
            self.source_galaxies.source_0 = phase2_results.variable.source_0
            self.source_galaxies.source_1 = phase2_results.variable.source_1
            self.source_galaxies.source_0.hyper_galaxy = phase2h_results.constant.source_0.hyper_galaxy
            self.source_galaxies.source_1.hyper_galaxy = phase2h_results.constant.source_1.hyper_galaxy

    phase3 = BothSourceGalaxiesPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source_0=gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                          source_1=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest, phase_name="{}/phase3".format(test_name))

    return pl.PipelineImaging(test_name, phase1, phase1h, phase2, phase2h, phase3)


if __name__ == "__main__":
    pipeline()
