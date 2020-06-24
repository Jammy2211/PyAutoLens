import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "phase_features"
test_name = "positions"
data_name = "lens_sis__source_smooth"
instrument = "vro"


def make_pipeline(name, folders, search=af.DynestyStatic()):
    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        positions_threshold=0.0,
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 30
    phase1.search.facc = 0.8

    return al.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(
        sys.modules[__name__],
        positions=[[(1.6, 0.0), (0.0, 1.6), (-1.6, 0.0), (0.0, -1.6)]],
    )
