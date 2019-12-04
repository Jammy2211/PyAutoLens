import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_magnification__hyper"
data_type = "lens_mass__source_smooth"
data_resolution = "lsst"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    class SourcePix(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.mass.centre.centre_0 = 0.0
            self.galaxies.lens.mass.centre.centre_1 = 0.0
            self.galaxies.lens.mass.einstein_radius = 1.6

    phase1 = SourcePix(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    phase1.extend_with_multiple_hyper_phases(hyper_galaxy=True, inversion=True)

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase1.result.model.galaxies.source.pixelization,
                regularization=phase1.result.model.galaxies.source.regularization,
                hyper_galaxy=phase1.result.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    phase2.extend_with_multiple_hyper_phases(hyper_galaxy=True, inversion=True)

    phase3 = al.PhaseImaging(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=af.last[-1].model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase1.result.model.galaxies.source.pixelization,
                regularization=phase1.result.model.galaxies.source.regularization,
                hyper_galaxy=phase2.result.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 40
    phase3.optimizer.sampling_efficiency = 0.8

    phase3.extend_with_multiple_hyper_phases(hyper_galaxy=True, inversion=True)

    return al.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
