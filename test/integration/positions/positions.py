from autolens.pipeline import pipeline as pl
from autolens.pipeline import phase as ph
from autolens.profiles import mass_profiles as mp
from autolens.lensing import galaxy_model as gp
from autolens.autofit import non_linear as nl
from autolens.lensing import galaxy
from autolens import conf
from test.integration import tools

import numpy as np
import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '/gpfs/data/pdtw24/Lens/int/positions/'

def test_positions_pipeline():

    pipeline_name = "pos"

    conf.instance.output_path = output_path

    try:
        shutil.rmtree(output_path + pipeline_name)
    except FileNotFoundError:
        pass

    pipeline = make_positions_pipeline(pipeline_name=pipeline_name)

    results = pipeline.run(positions=[[[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]], pixel_scale=0.05)
    for result in results:
        print(result)

def make_positions_pipeline(pipeline_name):

    phase1 = ph.PhasePositions(lens_galaxies=[gp.GalaxyModel(sis=mp.SphericalIsothermal)],
                               optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.1

    return pl.PipelinePositions(pipeline_name, phase1)

if __name__ == "__main__":
    test_positions_pipeline()