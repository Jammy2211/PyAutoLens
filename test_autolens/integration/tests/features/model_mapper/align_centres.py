import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "model_mapper"
test_name = "align_centres"
data_type = "lens_light_dev_vaucouleurs"
data_resolution = "lsst"


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):
    class MMPhase(al.PhaseImaging):
        pass

    phase1 = MMPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light_0=al.lp.EllipticalSersic,
                light_1=al.lp.EllipticalSersic,
                mass=al.mp.EllipticalIsothermal,
                align_centres=True,
            )
        ),
        non_linear_class=non_linear_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    return al.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
