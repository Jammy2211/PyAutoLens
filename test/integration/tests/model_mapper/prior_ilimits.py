import os
import shutil

from autofit import conf
from autofit.core import non_linear as nl
from autofit.core import model_mapper as mm
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '{}/../output/model_mapper'.format(dirpath)

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

def pipeline():

    pipeline_name = "prior_limits"
    data_name = '/prior_limits'

    tools.reset_paths(data_name, pipeline_name, output_path)

    sersic = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=1.0, effective_radius=1.3,
                                 sersic_index=3.0)

    lens_galaxy = galaxy.Galaxy(light_profile=sersic)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.5, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[], target_signal_to_noise=10.0)
    image = tools.load_image(data_name=data_name, pixel_scale=0.5)

    cf_pipeline = make_pipeline(pipeline_name=pipeline_name)

    results = cf_pipeline.run(image=image)
    for result in results:
        print(result)


def make_pipeline(pipeline_name):
    class MMPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):
            self.lens_galaxies.lens.sersic.centre_0 = 0.0
            self.lens_galaxies.lens.sersic.centre_1 = 0.0
            self.lens_galaxies.lens.sersic.axis_ratio = mm.UniformPrior(lower_limit=-0.5, upper_limit=0.1)
            self.lens_galaxies.lens.sersic.phi = 90.0
            self.lens_galaxies.lens.sersic.intensity = mm.UniformPrior(lower_limit=-0.5, upper_limit=0.1)
            self.lens_galaxies.lens.sersic.effective_radius = 1.3
            self.lens_galaxies.lens.sersic.sersic_index = 3.0

    phase1 = MMPhase(lens_galaxies=dict(lens=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                     optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    class MMPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):
            print(self.lens_galaxies.lens.sersic.intensity.lower_limit)
            print(self.lens_galaxies.lens.sersic.intensity.upper_limit)
            self.lens_galaxies.lens.sersic.intensity = previous_results[0].variable.lens.sersic.intensity
            print(self.lens_galaxies.lens.sersic.intensity.lower_limit)
            print(self.lens_galaxies.lens.sersic.intensity.upper_limit)
            self.lens_galaxies.lens = previous_results[0].variable.lens
            print(self.lens_galaxies.lens.sersic.intensity.lower_limit)
            print(self.lens_galaxies.lens.sersic.intensity.upper_limit)

    phase2 = MMPhase(lens_galaxies=dict(lens=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                     optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(pipeline_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
