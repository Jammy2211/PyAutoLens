import os

import autofit as af
import autolens as al
from test_autolens.integration.tests.imaging import runner

test_type = "features"
test_name = "agg_dataset"
data_type = "lens_light_dev_vaucouleurs"
data_resolution = "lsst"


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):
    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, sersic=al.lp.EllipticalSersic)),
        non_linear_class=non_linear_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return al.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])

    test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = test_path + "../output"
    agg = af.Aggregator(directory=str(output_path))

    datasets = agg.dataset

    # This should print "test_dataset" -> see integration/tests/imaging/runner.py

    print(datasets[0].name)

    # This should print a dictionary {"test" : 2} -> see integration/tests/imaging/runner.py

    print(datasets[0].metadata)
