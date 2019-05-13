import os

from autofit import conf
from autofit.mapper import prior
from autofit.optimize import non_linear as nl
from autofit.tools import phase as autofit_ph
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'grid_search'
test_name = "multinest_grid_subhalo_parallel"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='no_lens_light_and_source_smooth', data_resolution='LSST')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):
    
    class QuickPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass.centre_0 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.mass.centre_1 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.lens_galaxies.lens.mass.axis_ratio = prior.UniformPrior(lower_limit=0.65, upper_limit=0.75)
            self.lens_galaxies.lens.mass.phi = prior.UniformPrior(lower_limit=40.0, upper_limit=50.0)
            self.lens_galaxies.lens.mass.einstein_radius = prior.UniformPrior(lower_limit=1.55, upper_limit=1.65)

            self.source_galaxies.source.light.centre_0 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.source_galaxies.source.light.centre_1 = prior.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
            self.source_galaxies.source.light.axis_ratio = prior.UniformPrior(lower_limit=0.75, upper_limit=0.85)
            self.source_galaxies.source.light.phi = prior.UniformPrior(lower_limit=50.0, upper_limit=70.0)
            self.source_galaxies.source.light.intensity = prior.UniformPrior(lower_limit=0.35, upper_limit=0.45)
            self.source_galaxies.source.light.effective_radius = prior.UniformPrior(lower_limit=0.45, upper_limit=0.55)
            self.source_galaxies.source.light.sersic_index = prior.UniformPrior(lower_limit=0.9, upper_limit=1.1)

    phase1 = QuickPhase(
        phase_name='phase_1', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
        optimizer_class=nl.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class GridPhase(autofit_ph.as_grid_search(ph.LensSourcePlanePhase, parallel=True)):

        @property
        def grid_priors(self):
            return [self.variable.lens_galaxies.subhalo.mass.centre_0,
                    self.variable.lens_galaxies.subhalo.mass.centre_1]

        def pass_priors(self, results):

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1').\
                constant.lens_galaxies.lens

            ### Lens Subhalo, Adjust priors to physical masses (10^6 - 10^10) and concentrations (6-24)

            self.lens_galaxies.subhalo.mass.kappa_s = prior.UniformPrior(lower_limit=0.0005, upper_limit=0.2)
            self.lens_galaxies.subhalo.mass.scale_radius = prior.UniformPrior(lower_limit=0.001, upper_limit=1.0)
            self.lens_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.lens_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

            ### Source Light, Sersic -> Sersic ###

            self.source_galaxies.source.light.centre = results.from_phase('phase_1').\
                variable_absolute(a=0.05).source_galaxies.source.light.centre

            self.source_galaxies.source.light.intensity = results.from_phase('phase_1').\
                variable_relative(r=0.5).source_galaxies.source.light.intensity

            self.source_galaxies.source.light.effective_radius = results.from_phase('phase_1').\
                variable_relative(r=0.5).source_galaxies.source.light.effective_radius

            self.source_galaxies.source.light.sersic_index = results.from_phase('phase_1').\
                variable_relative(r=0.5).source_galaxies.source.light.sersic_index

            self.source_galaxies.source.light.axis_ratio = results.from_phase('phase_1').\
                variable_absolute(a=0.1).source_galaxies.source.light.axis_ratio

            self.source_galaxies.source.light.phi = results.from_phase('phase_1').\
                variable_absolute(a=20.0).source_galaxies.source.light.phi


    phase2 = GridPhase(
        phase_name='phase_2', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal),
                           subhalo=gm.GalaxyModel(mass=mp.SphericalTruncatedNFWChallenge)),
        source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
        optimizer_class=nl.MultiNest, number_of_steps=2)

    phase2.optimizer.const_efficiency_mode = True

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
