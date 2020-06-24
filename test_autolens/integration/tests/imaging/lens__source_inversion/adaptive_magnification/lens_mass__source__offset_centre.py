import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_magnification__offset_centre"
data_name = "lens_sie__source_smooth__offset_centre"
instrument = "vro"


def make_pipeline(name, folders, search=af.DynestyStatic()):

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    mass.centre.centre_0 = 2.0
    mass.centre.centre_1 = 2.0
    mass.einstein_radius = 1.6

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=mass),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
            ),
        ),
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 60
    phase1.search.facc = 0.8

    phase1 = phase1.extend_with_inversion_phase()

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase1.result.inversion.instance.galaxies.source.pixelization,
                regularization=phase1.result.inversion.instance.galaxies.source.regularization,
            ),
        ),
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 60
    phase2.search.facc = 0.8

    phase2 = phase2.extend_with_inversion_phase()

    return al.PipelineDataset(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
