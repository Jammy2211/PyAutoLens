import os

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_and_source_inversion'
test_name = "lens_both_x1_source_x1_adaptive_weighted"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='lens_light_and_source_smooth', data_resolution='LSST')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)


def make_pipeline(test_name):

    phase1 = phase_imaging.LensSourcePlanePhase(
        phase_name='phase_1', phase_folders=[test_type, test_name],
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalSersic)),
        optimizer_class=af.MultiNest)

    class InversionPhase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1').constant.lens_galaxies.lens

    phase2 = InversionPhase(
        phase_name='phase_2_weighted_regularization', phase_folders=[test_type, test_name],
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiBrightnessImage,
                regularization=reg.AdaptiveBrightness)),
        inversion_pixel_limit=50,
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_inversion_phase()
    
    class InversionPhase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1').\
                variable.lens_galaxies.lens
            
            self.source_galaxies.source = results.last.inversion.\
                constant.source_galaxies.source

    phase3 = InversionPhase(
        phase_name='phase_3', phase_folders=[test_type, test_name],
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiBrightnessImage,
                regularization=reg.AdaptiveBrightness)),
        inversion_pixel_limit=50,
        optimizer_class=af.MultiNest)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8

    phase3 = phase3.extend_with_inversion_phase()

    return pl.PipelineImaging(test_name, phase1, phase2, phase3)


if __name__ == "__main__":
    pipeline()
