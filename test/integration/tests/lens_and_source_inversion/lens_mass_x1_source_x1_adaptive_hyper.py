import os

import autofit as af
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as pix, regularization as reg
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline import pipeline as pl
from autolens.pipeline.phase import phase_imaging, phase_extensions
from test.integration import integration_util
from test.simulation import simulation_util

test_type = 'lens_and_source_inversion'
test_name = "lens_mass_x1_source_x1_adaptive_hyper"

test_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + 'output/'
config_path = test_path + 'config'
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


def pipeline():
    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(
        data_type='no_lens_light_and_source_smooth', data_resolution='LSST')
    pipeline = make_pipeline(test_name=test_name)
    results = pipeline.run(data=ccd_data)

    assert hasattr(results.last, "hyper")


def make_pipeline(test_name):

    class SourcePix(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass.centre.centre_0 = 0.0
            self.lens_galaxies.lens.mass.centre.centre_1 = 0.0
            self.lens_galaxies.lens.mass.einstein_radius = 1.6
            self.source_galaxies.source.pixelization.shape.shape_0 = 20.0
            self.source_galaxies.source.pixelization.shape.shape_1 = 20.0

    phase1 = SourcePix(
        phase_name='phase_1', phase_folders=[test_type, test_name],
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(
            source=gm.GalaxyModel(redshift=1.0, pixelization=pix.VoronoiMagnification,
                                  regularization=reg.Constant)),
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    phase1h = phase_extensions.HyperGalaxyPhase(phase_name='phase_1_hyper',
                                                phase_folders=[test_type, test_name])

    class HyperLensSourcePlanePhase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.hyper_galaxy = results.from_phase('phase_1_hyper'). \
                constant.lens_galaxies.lens.hyper_galaxy

            self.lens_galaxies.lens.mass = results.from_phase('phase_1'). \
                variable.lens_galaxies.lens.mass

            self.source_galaxies.source.hyper_galaxy = results.from_phase('phase_1_hyper'). \
                constant.source_galaxies.source.hyper_galaxy

            self.source_galaxies.source.pixelization = results.from_phase('phase_1'). \
                variable.source_galaxies.source.pixelization

            self.source_galaxies.source.regularization = results.from_phase('phase_1'). \
                variable.source_galaxies.source.regularization

    phase2 = HyperLensSourcePlanePhase(
        phase_name='phase_2', phase_folders=[test_type, test_name],
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal,
                                hyper_galaxy=g.HyperGalaxy)),
        source_galaxies=dict(
            source=gm.GalaxyModel(redshift=1.0, pixelization=pix.VoronoiMagnification,
                                  regularization=reg.Constant,
                                  hyper_galaxy=g.HyperGalaxy)),
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase1h, phase2)


if __name__ == "__main__":
    pipeline()
