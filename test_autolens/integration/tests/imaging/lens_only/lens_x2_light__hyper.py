import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens_only"
test_name = "lens_x2_light__hyper"
data_type = "lens_x2_light"
data_resolution = "lsst"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    def modify_mask_function(shape_2d, pixel_scales):
        return al.mask.circular(
            shape_2d=shape_2d, pixel_scales=pixel_scales, radius=5.0
        )

    class LensPlaneGalaxyX2Phase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens_0.light.centre_0 = -1.0
            self.galaxies.lens_0.light.centre_1 = -1.0

            self.galaxies.lens_1.light.centre_0 = 1.0
            self.galaxies.lens_1.light.centre_1 = 1.0

    phase1 = LensPlaneGalaxyX2Phase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
            lens_1=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
        ),
        mask_function=modify_mask_function,
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(hyper_galaxy=True)

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens_0=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.model.galaxies.lens_0.light,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.lens_0.hyper_galaxy,
            ),
            lens_1=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.model.galaxies.lens_1.light,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.lens_1.hyper_galaxy,
            ),
        ),
        mask_function=modify_mask_function,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    return al.PipelineDataset(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
