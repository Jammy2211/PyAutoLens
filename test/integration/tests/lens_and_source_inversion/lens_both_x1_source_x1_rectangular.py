import os

import autofit as af
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_and_source_inversion'
test_name = "lens_both_x1_source_x1"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


def pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(data_type='lens_and_source_smooth', data_resolution='Euclid')
    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)

def make_pipeline(test_name):

    class SourcePix(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass.centre.centre_0 = 0.0
            self.lens_galaxies.lens.mass.centre.centre_1 = 0.0
            self.lens_galaxies.lens.mass.einstein_radius_in_units = 1.6
            self.source_galaxies.source.pixelization.shape_0 = 20.0
            self.source_galaxies.source.pixelization.shape_1 = 20.0

    phase1 = SourcePix(
        phase_name='phase_1', phase_folders=[test_type, test_name],
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, light=lp.SphericalDevVaucouleurs, mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(source=gm.GalaxyModel(redshift=1.0, pixelization=pix.Rectangular, regularization=reg.Constant)),
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1)


if __name__ == "__main__":
    pipeline()
