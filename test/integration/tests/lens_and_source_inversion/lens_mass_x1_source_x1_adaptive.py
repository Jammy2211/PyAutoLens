import os

from autofit import conf
from autofit.core import non_linear as nl
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.integration import tools

test_type = 'lens_and_source_inversion'
test_name = "lens_mass_x1_source_x1_adaptive"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    lens_mass = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=80.0, einstein_radius=1.6)
    source_light = lp.EllipticalSersic(centre=(-0.0, 0.0), axis_ratio=0.6, phi=90.0, intensity=1.0,
                                       effective_radius=0.5, sersic_index=1.0)

    lens_galaxy = galaxy.Galaxy(sie=lens_mass)
    source_galaxy = galaxy.Galaxy(sersic=source_light)

    tools.reset_paths(test_name=test_name, output_path=output_path)
    tools.simulate_integration_image(test_name=test_name, pixel_scale=0.1, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[], target_signal_to_noise=30.0)
    image = tools.load_image(test_name=test_name, pixel_scale=0.1)

    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=image)


def make_pipeline(test_name):
    class SourcePix(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):
            self.lens_galaxies[0].sie.centre.centre_0 = 0.0
            self.lens_galaxies[0].sie.centre.centre_1 = 0.0
            self.lens_galaxies[0].sie.einstein_radius = 1.6
            self.source_galaxies[0].pixelization.shape_0 = 20.0
            self.source_galaxies[0].pixelization.shape_1 = 20.0

    phase1 = SourcePix(lens_galaxies=[gm.GalaxyModel(sie=mp.EllipticalIsothermal)],
                       source_galaxies=[gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                       regularization=reg.Constant)],
                       optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(test_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
