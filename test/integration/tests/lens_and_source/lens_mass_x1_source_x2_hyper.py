import os
import shutil

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_and_source'
test_name = "lens_mass_x1_source_x2_hyper"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_resolution='LSST', data_name='no_lens_light_and_source_smooth')
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
