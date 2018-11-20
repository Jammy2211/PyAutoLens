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


def test_lens_x1_src_x1_profile_hyper_pipeline():
    pipeline_name = "lens_x1_source_x2_hyper"
    data_name = '/lens_x1_source_x2_hyper'

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

    pipeline = make_lens_x1_src_x1_profile_hyper_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)


def make_lens_x1_src_x1_profile_hyper_pipeline(pipeline_name):
    phase1 = ph.LensSourcePlanePhase(lens_galaxies=[gm.GalaxyModel(sie=mp.EllipticalIsothermal)],
                                     source_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                     optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.7

    phase1h = ph.LensMassAndSourceProfileHyperOnlyPhase(optimizer_class=nl.MultiNest,
                                                        phase_name="{}/phase1h".format(pipeline_name))

    class AddSourceGalaxyPhase(ph.LensSourcePlaneHyperPhase):
        def pass_priors(self, previous_results):
            phase1_results = previous_results[-1]
            phase1h_results = previous_results[-1].hyper
            self.lens_galaxies[0] = phase1_results.variable.lens_galaxies[0]
            self.source_galaxies[0] = phase1_results.variable.source_galaxies[0]
            self.source_galaxies[0].hyper_galaxy = phase1h_results.constant.source_galaxies[0].hyper_galaxy

    phase2 = AddSourceGalaxyPhase(lens_galaxies=[gm.GalaxyModel(sie=mp.EllipticalIsothermal)],
                                  source_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                   gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                  optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.7

    phase2h = ph.LensMassAndSourceProfileHyperOnlyPhase(optimizer_class=nl.MultiNest,
                                                        phase_name="{}/phase1h".format(pipeline_name))

    class BothSourceGalaxiesPhase(ph.LensSourcePlaneHyperPhase):
        def pass_priors(self, previous_results):
            phase2_results = previous_results[1]
            phase2h_results = previous_results[1].hyper
            self.lens_galaxies[0] = phase2_results.variable.lens_galaxies[0]
            self.source_galaxies[0] = phase2_results.variable.source_galaxies[0]
            self.source_galaxies[1] = phase2_results.variable.source_galaxies[1]
            self.source_galaxies[0].hyper_galaxy = phase2h_results.constant.source_galaxies[0].hyper_galaxy
            self.source_galaxies[1].hyper_galaxy = phase2h_results.constant.source_galaxies[1].hyper_galaxy

    phase3 = BothSourceGalaxiesPhase(lens_galaxies=[gm.GalaxyModel(sie=mp.EllipticalIsothermal)],
                                     source_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                      gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                     optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(pipeline_name))

    return pl.PipelineImaging(pipeline_name, phase1, phase1h, phase2, phase2h, phase3)


if __name__ == "__main__":
    test_lens_x1_src_x1_profile_hyper_pipeline()
