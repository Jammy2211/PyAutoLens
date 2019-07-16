import os

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.pipeline.phase import phase_imaging, phase_extensions
from autolens.pipeline import pipeline as pl
from test.integration import integration_util
from test.simulation import simulation_util

test_type = "lens_and_source"
test_name = "lens_mass_x1_source_x2_hyper"

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)


def run_pipeline():

    integration_util.reset_paths(test_name=test_name, output_path=output_path)
    ccd_data = simulation_util.load_test_ccd_data(
        data_type="no_lens_light_and_source_smooth", data_resolution="LSST"
    )
    pipeline = make_pipeline()
    pipeline.run(data=ccd_data)


def make_pipeline():

    phase1 = phase_imaging.LensSourcePlanePhase(
        phase_name="phase_1",
        phase_folders=[test_type, test_name],
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)
        ),
        source_galaxies=dict(
            source_0=gm.GalaxyModel(redshift=1.0, sersic=lp.EllipticalSersic)
        ),
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.7

    class AddSourceGalaxyPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase(
                "phase_1"
            ).variable.lens_galaxies.lens
            self.source_galaxies.source_0 = results.from_phase(
                "phase_1"
            ).variable.source_galaxies.source_0

    phase2 = AddSourceGalaxyPhase(
        phase_name="phase_2",
        phase_folders=[test_type, test_name],
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)
        ),
        source_galaxies=dict(
            source_0=gm.GalaxyModel(redshift=1.0, sersic=lp.EllipticalSersic),
            source_1=gm.GalaxyModel(redshift=1.0, sersic=lp.EllipticalSersic),
        ),
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.7

    phase2 = phase2.extend_with_multiple_hyper_phases(hyper_galaxy=True)

    class HyperLensSourcePlanePhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase(
                "phase_1"
            ).variable.lens_galaxies.lens

            self.source_galaxies = results.from_phase(
                "phase_1"
            ).variable.source_galaxies

            self.source_galaxies.source_0.hyper_galaxy = (
                results.last.hyper_combined.constant.source_galaxies.source_0.hyper_galaxy
            )

            self.source_galaxies.source_1.hyper_galaxy = (
                results.last.hyper_combined.constant.source_galaxies.source_1.hyper_galaxy
            )

    phase3 = HyperLensSourcePlanePhase(
        phase_name="phase_3",
        phase_folders=[test_type, test_name],
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5, mass=mp.EllipticalIsothermal, hyper_galaxy=g.HyperGalaxy
            )
        ),
        source_galaxies=dict(
            source_0=gm.GalaxyModel(
                redshift=1.0, light=lp.EllipticalSersic, hyper_galaxy=g.HyperGalaxy
            ),
            source_1=gm.GalaxyModel(
                redshift=1.0, light=lp.EllipticalSersic, hyper_galaxy=g.HyperGalaxy
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase2, phase3)


if __name__ == "__main__":
    run_pipeline()
