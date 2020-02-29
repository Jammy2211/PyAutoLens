import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "grid_search"
test_name = "multinest_grid_subhalo__parallel"
data_type = "lens_sie__source_smooth"
data_resolution = "lsst"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class GridPhase(af.as_grid_search(al.PhaseImaging, parallel=True)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

        def customize_priors(self, results):

            ### Lens Subhalo, Adjust priors to physical masses (10^6 - 10^10) and concentrations (6-24)

            self.galaxies.subhalo.mass.kappa_s = af.UniformPrior(
                lower_limit=0.0005, upper_limit=0.2
            )
            self.galaxies.subhalo.mass.centre_0 = af.UniformPrior(
                lower_limit=-2.0, upper_limit=2.0
            )
            self.galaxies.subhalo.mass.centre_1 = af.UniformPrior(
                lower_limit=-2.0, upper_limit=2.0
            )

    phase1 = GridPhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            subhalo=al.GalaxyModel(
                redshift=0.5, mass=al.mp.SphericalTruncatedNFWChallenge
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
            ),
        ),
        optimizer_class=optimizer_class,
        number_of_steps=2,
    )

    phase1.optimizer.const_efficiency_mode = True

    return al.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
