import os

from autofit import conf
from autofit.core import non_linear as nl
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '{}/../output/lens_and_source_inversion'.format(dirpath)

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

def pipeline():
    pipeline_name = "lens_mass_x1_source_x1_rectangular"
    data_name = '/lens_mass_x1_source_x1_rectangular'

    tools.reset_paths(data_name, pipeline_name, output_path)

    lens_mass = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=80.0, einstein_radius=1.6)
    source_light = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.6, phi=90.0, intensity=1.0,
                                       effective_radius=0.5, sersic_index=1.0)

    lens_galaxy = galaxy.Galaxy(sie=lens_mass)
    source_galaxy = galaxy.Galaxy(sersic=source_light)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[source_galaxy], target_signal_to_noise=30.0)

    pipeline = make_lens_x1_source_x1_inversion_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)


def make_lens_x1_source_x1_inversion_pipeline(pipeline_name):
    class SourcePix(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):
            self.lens_galaxies[0].sie.centre.centre_0 = 0.0
            self.lens_galaxies[0].sie.centre.centre_1 = 0.0
            self.lens_galaxies[0].sie.einstein_radius = 1.6
            self.source_galaxies[0].pixelization.shape_0 = 20.0
            self.source_galaxies[0].pixelization.shape_1 = 20.0

    phase1 = SourcePix(lens_galaxies=[gm.GalaxyModel(sie=mp.EllipticalIsothermal)],
                       source_galaxies=[gm.GalaxyModel(pixelization=pix.Rectangular, regularization=reg.Constant)],
                       optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(pipeline_name, phase1)


if __name__ == "__main__":
    pipeline()
