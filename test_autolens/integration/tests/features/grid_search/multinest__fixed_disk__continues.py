import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "grid_search"
test_name = "multinest__fixed_disk__continues"
data_type = "lens_light_dev_vaucouleurs"
data_resolution = "euclid"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class QuickPhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.bulge.centre_0 = af.UniformPrior(
                lower_limit=-0.01, upper_limit=0.01
            )
            self.galaxies.lens.bulge.centre_1 = af.UniformPrior(
                lower_limit=-0.01, upper_limit=0.01
            )
            self.galaxies.lens.bulge.axis_ratio = af.UniformPrior(
                lower_limit=0.79, upper_limit=0.81
            )
            self.galaxies.lens.bulge.phi = af.UniformPrior(
                lower_limit=-1.0, upper_limit=1.0
            )
            self.galaxies.lens.bulge.intensity = af.UniformPrior(
                lower_limit=0.99, upper_limit=1.01
            )
            self.galaxies.lens.bulge.effective_radius = af.UniformPrior(
                lower_limit=1.25, upper_limit=1.35
            )
            self.galaxies.lens.bulge.sersic_index = af.UniformPrior(
                lower_limit=3.95, upper_limit=4.05
            )

            self.galaxies.lens.disk.centre_0 = af.UniformPrior(
                lower_limit=-0.01, upper_limit=0.01
            )
            self.galaxies.lens.disk.centre_1 = af.UniformPrior(
                lower_limit=-0.01, upper_limit=0.01
            )
            self.galaxies.lens.disk.axis_ratio = af.UniformPrior(
                lower_limit=0.69, upper_limit=0.71
            )
            self.galaxies.lens.disk.phi = af.UniformPrior(
                lower_limit=-1.0, upper_limit=1.0
            )
            self.galaxies.lens.disk.intensity = af.UniformPrior(
                lower_limit=1.99, upper_limit=2.01
            )
            self.galaxies.lens.disk.effective_radius = af.UniformPrior(
                lower_limit=1.95, upper_limit=2.05
            )

    phase1 = QuickPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                bulge=al.lp.EllipticalSersic,
                disk=al.lp.EllipticalExponential,
            )
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class GridPhase(af.as_grid_search(al.PhaseImaging)):
        @property
        def grid_priors(self):
            return [self.model.lens.bulge.sersic_index]

        def customize_priors(self, results):

            self.galaxies.lens.disk = results.from_phase("phase_1").instance.lens.disk

            self.galaxies.lens.bulge.centre_0 = af.UniformPrior(
                lower_limit=-0.01, upper_limit=0.01
            )
            self.galaxies.lens.bulge.centre_1 = af.UniformPrior(
                lower_limit=-0.01, upper_limit=0.01
            )
            self.galaxies.lens.bulge.axis_ratio = af.UniformPrior(
                lower_limit=0.79, upper_limit=0.81
            )
            self.galaxies.lens.bulge.phi = af.UniformPrior(
                lower_limit=-1.0, upper_limit=1.0
            )
            self.galaxies.lens.bulge.intensity = af.UniformPrior(
                lower_limit=0.99, upper_limit=1.01
            )
            self.galaxies.lens.bulge.effective_radius = af.UniformPrior(
                lower_limit=1.25, upper_limit=1.35
            )

    phase2 = GridPhase(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                bulge=al.lp.EllipticalSersic, disk=al.lp.EllipticalExponential
            )
        ),
        number_of_steps=2,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 10
    phase2.optimizer.sampling_efficiency = 0.5

    class BestResultPhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.disk = results.from_phase("phase_1").model.lens.disk
            self.galaxies.lens.bulge = results.from_phase(
                "phase_2"
            ).best_result.model.lens.bulge

    phase3 = BestResultPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                bulge=al.lp.EllipticalSersic,
                disk=al.lp.EllipticalExponential,
            )
        ),
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8

    return al.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
