import os

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import mass_profiles as mp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_and_source_inversion'
test_name = "lens_mass_x1_source_x1_rectangular"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)

def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='no_lens_light_and_source_smooth', data_resolution='Euclid')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)

def make_pipeline(test_name):
    
    class SourcePix(ph.LensSourcePlanePhase):

        def pass_priors(self, results):
            
            self.lens_galaxies.lens.mass.centre.centre_0 = 0.0
            self.lens_galaxies.lens.mass.centre.centre_1 = 0.0
            self.lens_galaxies.lens.mass.einstein_radius = 1.6
            self.source_galaxies.source.pixelization.shape_0 = 20.0
            self.source_galaxies.source.pixelization.shape_1 = 20.0

    phase1 = SourcePix(phase_name='phase_1', phase_folders=[test_type, test_name],
                       lens_galaxies=dict(lens=gm.GalaxyModel(sie=mp.EllipticalIsothermal)),
                       source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.Rectangular, 
                                                                  regularization=reg.Constant)),
                       optimizer_class=nl.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
