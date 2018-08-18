from autolens.pipeline import pipeline as pl
from autolens.profiles import light_profiles as lp
from autolens.analysis import galaxy
from autolens.autopipe import non_linear as nl
from autolens import conf
from test.integration import tools

import numpy as np
import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
output_path = '/gpfs/data/pdtw24/Lens/integration/'

def test_lens_x1_gal_hyper_pipeline():

    pipeline_name = "lens_x1_gal_hyper"
    data_name = '/lens_x1_gal_hyper'

    try:
        shutil.rmtree(dirpath+'/data'+data_name)
    except FileNotFoundError:
        pass

    bulge = lp.EllipticalSersic(centre=(0.01, 0.01), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                 effective_radius=1.0, sersic_index=4.0)

    disk = lp.EllipticalSersic(centre=(0.01, 0.01), axis_ratio=0.6, phi=90.0, intensity=1.0,
                                 effective_radius=2.5, sersic_index=1.0)

    lens_galaxy = galaxy.Galaxy(bulge=bulge, disk=disk)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy],
                                     source_galaxies=[])

    conf.instance.output_path = output_path

    # try:
    #     shutil.rmtree(output_path + pipeline_name)
    # except FileNotFoundError:
    #     pass

    pipeline = make_lens_x1_gal_hyper_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)

def make_lens_x1_gal_hyper_pipeline(pipeline_name):

    from autolens.pipeline import phase as ph
    from autolens.analysis import galaxy_prior as gp
    from autolens.imaging import mask as msk
    from autolens.profiles import light_profiles, mass_profiles

    # 1) Lens Light : EllipticalSersic
    #    Mass: None
    #    Source: None
    #    Hyper Galaxy: None
    #    NLO: MultiNest
    #    Image : Observed Image
    #    Mask : Circle - 3.0"

    phase1 = ph.LensPlanePhase(lens_galaxies=[gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic)],
                               optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    phase1h = ph.LensPlaneHyperOnlyPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase1h".format(pipeline_name))

    class LensHyperPhase(ph.LensPlaneHyperPhase):
        def pass_priors(self, previous_results):
            phase1_results = previous_results[-1]
            phase1h_results = previous_results[-1].hyper
            self.lens_galaxies = phase1_results.variable.lens_galaxies
            self.lens_galaxies[0].hyper_galaxy = phase1h_results.constant.lens_galaxies[0].hyper_galaxy

    phase2 = LensHyperPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(pipeline_name))

    return pl.Pipeline(pipeline_name, phase1, phase1h, phase2)

if __name__ == "__main__":
    test_lens_x1_gal_hyper_pipeline()