import autofit as af
import autolens as al
from test_autolens.integration.tests.interferometer import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_rectangular"
data_label = "lens_sie__source_smooth"
instrument = "sma"


def make_pipeline(name, phase_folders, real_space_mask, non_linear_class=af.MultiNest):

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    mass.centre.centre_0 = 0.0
    mass.centre.centre_1 = 0.0
    mass.einstein_radius = 1.6

    pixelization = af.PriorModel(al.pix.Rectangular)

    pixelization.shape_0 = 20.0
    pixelization.shape_1 = 20.0

    phase1 = al.PhaseInterferometer(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=mass),
            source=al.GalaxyModel(
                redshift=1.0, pixelization=pixelization, regularization=al.reg.Constant
            ),
        ),
        real_space_mask=real_space_mask,
        non_linear_class=non_linear_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    return al.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
