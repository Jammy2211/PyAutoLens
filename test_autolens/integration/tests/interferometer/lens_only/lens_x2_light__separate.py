import autofit as af
import autolens as al
from test_autolens.integration.tests.interferometer import runner

test_type = "lens_only"
test_name = "lens_x2_light__separate"
data_type = "lens_x2_light"
data_resolution = "sma"


def make_pipeline(
    name,
    phase_folders,
    real_space_shape_2d=(100, 100),
    real_space_pixel_scales=(0.1, 0.1),
    optimizer_class=af.MultiNest,
):


    class LensPlaneGalaxy0Phase(al.PhaseInterferometer):
        def customize_priors(self, results):

            self.galaxies.lens_0.light.centre_0 = -1.0
            self.galaxies.lens_0.light.centre_1 = -1.0

    phase1 = LensPlaneGalaxy0Phase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)
        ),
        real_space_shape_2d=real_space_shape_2d,
        real_space_pixel_scales=real_space_pixel_scales,
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class LensPlaneGalaxy1Phase(al.PhaseInterferometer):
        def customize_priors(self, results):

            self.galaxies.lens_1.light.centre_0 = 1.0
            self.galaxies.lens_1.light.centre_1 = 1.0

    phase2 = LensPlaneGalaxy1Phase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=phase1.result.instance.galaxies.lens_0,
            lens_1=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
        ),
        real_space_shape_2d=real_space_shape_2d,
        real_space_pixel_scales=real_space_pixel_scales,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    class LensPlaneBothGalaxyPhase(al.PhaseInterferometer):
        def customize_priors(self, results):

            self.galaxies.lens_0.light.centre_0 = -1.0
            self.galaxies.lens_0.light.centre_1 = -1.0
            self.galaxies.lens_1.light.centre_0 = 1.0
            self.galaxies.lens_1.light.centre_1 = 1.0

    phase3 = LensPlaneBothGalaxyPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=phase1.result.model.galaxies.lens_0,
            lens_1=phase2.result.model.galaxies.lens_1,
        ),
        real_space_shape_2d=real_space_shape_2d,
        real_space_pixel_scales=real_space_pixel_scales,
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return al.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
