import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "grid_search"
test_name = "normal__sersic"
data_type = "lens_light_dev_vaucouleurs"
data_resolution = "euclid"


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):
    class QuickPhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.light.centre_0 = af.UniformPrior(
                lower_limit=-0.01, upper_limit=0.01
            )
            self.galaxies.lens.light.centre_1 = af.UniformPrior(
                lower_limit=-0.01, upper_limit=0.01
            )
            self.galaxies.lens.light.axis_ratio = af.UniformPrior(
                lower_limit=0.79, upper_limit=0.81
            )
            self.galaxies.lens.light.phi = af.UniformPrior(
                lower_limit=-1.0, upper_limit=1.0
            )
            self.galaxies.lens.light.intensity = af.UniformPrior(
                lower_limit=0.99, upper_limit=1.01
            )
            self.galaxies.lens.light.effective_radius = af.UniformPrior(
                lower_limit=1.25, upper_limit=1.35
            )
            self.galaxies.lens.light.sersic_index = af.UniformPrior(
                lower_limit=3.95, upper_limit=4.05
            )

    phase1 = QuickPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)),
        non_linear_class=non_linear_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class GridPhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.light.centre_0 = 0.0
            self.galaxies.lens.light.centre_1 = 0.0
            self.galaxies.lens.light.axis_ratio = results.from_phase(
                "phase_1"
            ).instance.lens.light.axis_ratio
            self.galaxies.lens.light.phi = results.from_phase(
                "phase_1"
            ).instance.lens.light.phi
            self.galaxies.lens.light.intensity = results.from_phase(
                "phase_1"
            ).instance.lens.light.intensity

            self.galaxies.lens.light.effective_radius = af.UniformPrior(
                lower_limit=0.0, upper_limit=4.0
            )
            self.galaxies.lens.light.sersic_index = af.UniformPrior(
                lower_limit=1.0, upper_limit=8.0
            )

    phase2 = GridPhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)),
        non_linear_class=af.GridSearch,
    )

    phase2.optimizer.const_efficiency_mode = True

    return al.PipelineDataset(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
