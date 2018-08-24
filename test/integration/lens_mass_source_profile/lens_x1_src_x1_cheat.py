from autolens.pipeline import pipeline as pl
from autolens.pipeline import phase as ph
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.analysis import galaxy_prior as gp
from autolens.autopipe import non_linear as nl
from autolens.autopipe import model_mapper as mm
from autolens.analysis import galaxy
from autolens import conf
from test.integration import tools

import numpy as np
import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '/gpfs/data/pdtw24/Lens/int/lens_mass_source/'

def test_lens_x1_src_x1_profile_pipeline():

    pipeline_name = "l1_s1_c"
    data_name = '/l1_s1_c'

    try:
        shutil.rmtree(dirpath+'/data'+data_name)
    except FileNotFoundError:
        pass

    sersic = lp.EllipticalSersicLP(centre=(0.01, 0.01), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                   effective_radius=1.3, sersic_index=3.0)

    lens_mass = mp.EllipticalIsothermalMP(centre=(0.01, 0.01), axis_ratio=0.8, phi=80.0, einstein_radius=1.6)
    source_light = lp.EllipticalSersicLP(centre=(-0.01, -0.01), axis_ratio=0.6, phi=90.0, intensity=1.0,
                                         effective_radius=0.5, sersic_index=1.0)

    lens_galaxy = galaxy.Galaxy(sersic=sersic, sie=lens_mass)
    source_galaxy = galaxy.Galaxy(sersic=source_light)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.08, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[source_galaxy], target_signal_to_noise=30.0)

    conf.instance.output_path = output_path

    try:
        shutil.rmtree(output_path + pipeline_name)
    except FileNotFoundError:
        pass

    pipeline = make_lens_x1_src_x1_profile_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.08)

    results = pipeline.run(image=image)
    for result in results:
        print(result)

def make_lens_x1_src_x1_profile_pipeline(pipeline_name):

    phase0 = ph.LensMassAndSourceProfilePhase(lens_galaxies=[gp.GalaxyPrior(sie=mp.EllipticalIsothermalMP)],
                                              source_galaxies=[gp.GalaxyPrior(sersic=lp.EllipticalSersicLP)],
                                              optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1 = ph.LensMassAndSourceProfilePhase(lens_galaxies=[gp.GalaxyPrior(sie=mp.EllipticalIsothermalMP)],
                                              source_galaxies=[gp.GalaxyPrior(sersic=lp.EllipticalSersicLP)],
                                              optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(pipeline_name, phase0, phase1)


if __name__ == "__main__":
    test_lens_x1_src_x1_profile_pipeline()
