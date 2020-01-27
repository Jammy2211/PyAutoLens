import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_rectangular__offset_centre"
data_type = "lens_sie__source_smooth__offset_centre"
data_resolution = "euclid"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class SourcePix(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.mass.centre.centre_0 = 2.0
            self.galaxies.lens.mass.centre.centre_1 = 2.0
            self.galaxies.lens.mass.einstein_radius = 1.2
            self.galaxies.source.pixelization.shape_0 = 20.0
            self.galaxies.source.pixelization.shape_1 = 20.0

    phase1 = SourcePix(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.Rectangular,
                regularization=al.reg.Constant,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    return al.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
