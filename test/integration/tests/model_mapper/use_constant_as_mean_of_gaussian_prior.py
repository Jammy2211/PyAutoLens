import os

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'model_mapper'
test_name = "use_constant_as_mean_of_gaussian_prior"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='lens_only_dev_vaucouleurs', data_resolution='LSST')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):

    class MMPhase(phase_imaging.LensPlanePhase):

        pass

    phase1 = MMPhase(phase_name='phase_1', phase_folders=[test_name],
                     lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                     optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    class MMPhase2(phase_imaging.LensPlanePhase):

        def pass_priors(self, results):

            centre_value = results.from_phase('phase_1').constant.lens_galaxies.lens.light.centre
            self.lens_galaxies.lens.light.centre.centre_0 = af.prior.GaussianPrior(mean=centre_value[0],  sigma=0.5)
            self.lens_galaxies.lens.light.centre.centre_1 = af.prior.GaussianPrior(mean=centre_value[1],  sigma=0.5)

            intensity_value = results.from_phase('phase_1').constant.lens_galaxies.lens.light.intensity
            self.lens_galaxies.lens.light.intensity = af.prior.GaussianPrior(mean=intensity_value,  sigma=1.0)

            effective_radius_value = results.from_phase('phase_1').constant.lens_galaxies.lens.light.effective_radius
            self.lens_galaxies.lens.light.effective_radius = af.prior.GaussianPrior(mean=effective_radius_value,  sigma=2.0)

            sersic_index_value = results.from_phase('phase_1').constant.lens_galaxies.lens.light.sersic_index
            self.lens_galaxies.lens.light.sersic_index = af.prior.GaussianPrior(mean=sersic_index_value,  sigma=2.0)

            axis_ratio_value = results.from_phase('phase_1').constant.lens_galaxies.lens.light.axis_ratio
            self.lens_galaxies.lens.light.axis_ratio = af.prior.GaussianPrior(mean=axis_ratio_value,  sigma=0.3)

            phi_value = results.from_phase('phase_1').constant.lens_galaxies.lens.light.phi
            self.lens_galaxies.lens.light.phi = af.prior.GaussianPrior(mean=phi_value,  sigma=30.0)

    phase2 = MMPhase2(
        phase_name='phase_2', phase_folders=[test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic)),
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase2)


if __name__ == "__main__":
    pipeline()
