import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.pipeline import pipeline as pl
from autolens.pipeline.phase import phase_imaging
from test.integration.tests import runner

test_type = "lens__source"
test_name = "lens_mass__source_x2"
data_type = "lens_mass__source_smooth"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    phase1 = phase_imaging.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=al.EllipticalIsothermal),
            source_0=gm.GalaxyModel(redshift=1.0, sersic=al.EllipticalSersic),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.7

    phase2 = phase_imaging.PhaseImaging(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=phase1.result.variable.galaxies.lens,
            source_0=phase1.result.variable.galaxies.source_0,
            source_1=gm.GalaxyModel(redshift=1.0, sersic=al.EllipticalSersic),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.7

    return pl.PipelineImaging(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
