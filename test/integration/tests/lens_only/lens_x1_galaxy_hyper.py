import os

from autofit.core import non_linear as nl
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '{}/../output/lens_only'.format(dirpath)

def pipeline():

    pipeline_name = 'lens_x1_galaxy_hyper'
    data_name = '/lens_x1_galaxy_hyper'

    tools.reset_paths(data_name, pipeline_name, output_path)

    bulge_0 = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                  effective_radius=1.0, sersic_index=4.0)

    bulge_1 = lp.EllipticalSersic(centre=(0.4, 0.4), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                  effective_radius=1.0, sersic_index=4.0)

    lens_galaxy = galaxy.Galaxy(bulge_0=bulge_0, bulge_1=bulge_1)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[], target_signal_to_noise=30.0)

    pipeline = make_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)


def make_pipeline(pipeline_name):
    phase1 = ph.LensPlanePhase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                               optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    phase1h = ph.LensLightHyperOnlyPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase1h".format(pipeline_name))

    class LensHyperPhase(ph.LensPlaneHyperPhase):
        def pass_priors(self, previous_results):
            phase1_results = previous_results[-1]
            phase1h_results = previous_results[-1].hyper
            self.lens_galaxies = phase1_results.variable.lens_galaxies
            self.lens_galaxies[0].hyper_galaxy = phase1h_results.constant.lens_galaxies[0].hyper_galaxy

    phase2 = LensHyperPhase(lens_galaxies=[], optimizer_class=nl.MultiNest,
                            phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(pipeline_name, phase1, phase1h, phase2)


if __name__ == "__main__":
    pipeline()
