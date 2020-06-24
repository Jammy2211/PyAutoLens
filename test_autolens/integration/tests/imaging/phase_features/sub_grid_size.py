import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "phase_features"
test_name = "sub_size"
data_name = "lens_light_dev_vaucouleurs"
instrument = "vro"


def make_pipeline(name, folders, search=af.DynestyStatic()):

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, sersic=al.lp.EllipticalSersic)),
        sub_size=2,
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 40
    phase1.search.facc = 0.8

    return al.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
