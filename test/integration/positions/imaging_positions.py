import os
import shutil

from autolens import conf
from autolens.autofit import non_linear as nl
from autolens.lensing import galaxy
from autolens.lensing import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '/gpfs/data/pdtw24/Lens/int/positions/'


def test_lens_x1_gal_pipeline():
    pipeline_name = "img_pos"
    data_name = '/img_pos'

    try:
        shutil.rmtree(dirpath + '/data' + data_name)
    except FileNotFoundError:
        pass

    sersic = lp.EllipticalSersic(centre=(0.01, 0.01), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                 effective_radius=1.3, sersic_index=3.0)
    sis = mp.SphericalIsothermal(einstein_radius=1.0)

    lens_galaxy = galaxy.Galaxy(mass_profile=sis)
    source_galaxy = galaxy.Galaxy(light_profile=sersic)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.1, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[source_galaxy], target_signal_to_noise=30.0)

    conf.instance.output_path = output_path

    try:
        shutil.rmtree(output_path + pipeline_name)
    except FileNotFoundError:
        pass

    pipeline = make_imaging_positions_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.1)

    results = pipeline.run(image=image)
    for result in results:
        print(result)


def make_imaging_positions_pipeline(pipeline_name):
    phase1 = ph.PositionsImagingPhase(positions=[[[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]],
                                      lens_galaxies=[gm.GalaxyModel(sis=mp.SphericalIsothermal)],
                                      optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(pipeline_name, phase1)


if __name__ == "__main__":
    test_lens_x1_gal_pipeline()
