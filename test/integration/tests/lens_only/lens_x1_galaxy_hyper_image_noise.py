import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.hyper import hyper_data as hi
from autolens.model.profiles import light_profiles as lp
from test.integration.tests import runner

test_type = "lens_only"
test_name = "lens_x1_galaxy_hyper_image_noise"
data_type = "lens_only_dev_vaucouleurs"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):

    phase1 = phase_imaging.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic)),
        hyper_image_sky=hi.HyperImageSky,
        hyper_noise_background=hi.HyperNoiseBackground,
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
