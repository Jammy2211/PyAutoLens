import os

from autofit.core import non_linear as nl
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import mass_profiles as mp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '{}/../output/positions'.format(dirpath)


def positions_pipeline():

    data_name = ''
    pipeline_name = "positions_phase"

    tools.reset_paths(data_name, pipeline_name, output_path)

    pipeline = make_positions_pipeline(pipeline_name=pipeline_name)

    results = pipeline.run(positions=[[[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]], pixel_scale=0.05)
    for result in results:
        print(result)


def make_positions_pipeline(pipeline_name):
    phase1 = ph.PhasePositions(lens_galaxies=[gm.GalaxyModel(sis=mp.SphericalIsothermal)],
                               optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.1

    return pl.PipelinePositions(pipeline_name, phase1)


if __name__ == "__main__":
    positions_pipeline()
