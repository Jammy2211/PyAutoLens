import os

from autofit.core import non_linear as nl
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '{}/../output/model_mapper'.format(dirpath)


def pipeline():
    pipeline_name = "const_tuple_then_float"
    data_name = '/const_tuple_then_float'

    tools.reset_paths(data_name, pipeline_name, output_path)

    sersic = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=1.0, effective_radius=1.3,
                                 sersic_index=3.0)

    lens_galaxy = galaxy.Galaxy(light_profile=sersic)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.5, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[], target_signal_to_noise=10.0)
    image = tools.load_image(data_name=data_name, pixel_scale=0.5)

    pipeline = make_pipeline(pipeline_name=pipeline_name)

    results = pipeline.run(image=image)
    for result in results:
        print(result)


def make_pipeline(pipeline_name):
    class MMPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):
            self.lens_galaxies[0].sersic.centre_0 = 1.0
            self.lens_galaxies[0].sersic.centre_1 = 2.0
            self.lens_galaxies[0].sersic.axis_ratio = 0.2
            self.lens_galaxies[0].sersic.phi = 90.0

    phase1 = MMPhase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                     optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(pipeline_name, phase1)


if __name__ == "__main__":
    pipeline()
