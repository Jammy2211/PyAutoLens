import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from test.simulation import simulation_util

test_type = "grid_search"
test_name = "multinest_grid_fixed_disk_parallel"
data_type = "lens_only_dev_vaucouleurs"
data_resolution = "Euclid"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class QuickPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

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
            lens=gm.GalaxyModel(
                redshift=0.5, bulge=lp.EllipticalSersic, disk=lp.EllipticalExponential
            )
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class GridPhase(
        af.as_grid_search(phase_class=phase_imaging.PhaseImaging, parallel=True)
    ):
        @property
        def grid_priors(self):
            return [self.variable.galaxies.lens.bulge.sersic_index]

        def pass_priors(self, results):

            self.galaxies.lens.disk = results.from_phase(
                "phase_1"
            ).constant.galaxies.lens.disk

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
            lens=gm.GalaxyModel(
                redshift=0.5, bulge=lp.EllipticalSersic, disk=lp.EllipticalExponential
            )
        ),
        number_of_steps=6,
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True

    return pl.PipelineImaging(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
