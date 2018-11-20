import os
import shutil

from autofit import conf
from autofit.core import non_linear as nl
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '{}/../output/lens_and_source'.format(dirpath)


def test_lens_x1_src_x1_profile_pipeline():
    pipeline_name = "lens_x1_source_x2"
    data_name = '/lens_x1_source_x2'

    try:
        shutil.rmtree(dirpath + '/datas' + data_name)
    except FileNotFoundError:
        pass

    lens_mass = mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=80.0, einstein_radius=1.6)
    source_light_0 = lp.EllipticalSersic(centre=(-0.6, 0.5), axis_ratio=0.6, phi=60.0, intensity=1.0,
                                         effective_radius=0.5, sersic_index=1.0)
    source_light_1 = lp.EllipticalSersic(centre=(0.2, 0.3), axis_ratio=0.6, phi=90.0, intensity=1.0,
                                         effective_radius=0.5, sersic_index=1.0)

    lens_galaxy = galaxy.Galaxy(sie=lens_mass)
    source_galaxy_0 = galaxy.Galaxy(sersic=source_light_0)
    source_galaxy_1 = galaxy.Galaxy(sersic=source_light_1)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[source_galaxy_0, source_galaxy_1], target_signal_to_noise=30.0)

    conf.instance.output_path = output_path

    # try:
    #     shutil.rmtree(output_path + pipeline_name)
    # except FileNotFoundError:
    #     pass

    pipeline = make_lens_x1_src_x1_profile_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)


def make_lens_x1_src_x1_profile_pipeline(pipeline_name):
    phase1 = ph.LensSourcePlanePhase(lens_galaxies=[gm.GalaxyModel(sie=mp.EllipticalIsothermal)],
                                     source_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                     optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.7

    phase1 = ph.LensSourcePlanePhase(lens_galaxies=[gm.GalaxyModel(sie=mp.EllipticalIsothermal)],
                                     source_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                     optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    class AddSourceGalaxyPhase(ph.LensSourcePlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0] = previous_results[0].variable.lens_galaxies[0]
            self.source_galaxies[0] = previous_results[0].variable.source_galaxies[0]

    phase2 = AddSourceGalaxyPhase(lens_galaxies=[gm.GalaxyModel(sie=mp.EllipticalIsothermal)],
                                  source_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                   gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                  optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.7

    return pl.PipelineImaging(pipeline_name, phase1, phase2)


if __name__ == "__main__":
    test_lens_x1_src_x1_profile_pipeline()
