import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "grid_search"
test_name = "multinest_grid_subhalo__hyper__parallel"
data_type = "lens_sie__source_smooth"
data_resolution = "lsst"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        optimizer_class=optimizer_class,
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=phase1.result.instance.galaxies.lens,
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=False, inversion=False
    )

    class GridPhase(af.as_grid_search(al.PhaseImaging, parallel=True)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    subhalo = al.GalaxyModel(
        redshift=0.5, mass=al.mp.SphericalTruncatedNFWMassToConcentration
    )

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)

    subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    phase3 = GridPhase(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase2.result.model.galaxies.lens.mass
            ),
            subhalo=subhalo,
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
            ),
        ),
        optimizer_class=optimizer_class,
        number_of_steps=2,
    )

    phase3.optimizer.const_efficiency_mode = True

    return al.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
