from autolens.pipeline import pipeline as pl
from autolens.profiles import light_profiles as lp
from autolens.autopipe import non_linear as nl
from autolens.autopipe import model_mapper
from autolens.analysis import galaxy
from autolens import conf
from test.integration import tools

import numpy as np
import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
output_path = '/gpfs/data/pdtw24/Lens/integration/'

def test_lens_x2_gal_pipeline():

    pipeline_name = "lens_x2_gal"
    data_name = '/lens_x2_gal'

    try:
        shutil.rmtree(dirpath+'/data'+data_name)
    except FileNotFoundError:
        pass

    sersic_0 = lp.EllipticalSersic(centre=(-2.0, -2.0), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                 effective_radius=1.3, sersic_index=3.0)

    sersic_1 = lp.EllipticalSersic(centre=( 2.0,  2.0), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                 effective_radius=1.3, sersic_index=3.0)

    lens_galaxy_0 = galaxy.Galaxy(light_profile=sersic_0)
    lens_galaxy_1 = galaxy.Galaxy(light_profile=sersic_1)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy_0, lens_galaxy_1],
                                     source_galaxies=[])

    conf.instance.output_path = output_path

    try:
        shutil.rmtree(output_path + pipeline_name)
    except FileNotFoundError:
        pass

    pipeline = make_lens_x2_gal_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)

def make_lens_x2_gal_pipeline(pipeline_name):

    from autolens.pipeline import phase as ph
    from autolens.analysis import galaxy_prior as gp
    from autolens.imaging import mask as msk
    from autolens.profiles import light_profiles

    class LensPlanex2GalPhase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0].elliptical_sersic.centre.centre_0 = model_mapper.UniformPrior(-3.0, -1.0)
            self.lens_galaxies[0].elliptical_sersic.centre.centre_1 = model_mapper.UniformPrior(-3.0, -1.0)
            self.lens_galaxies[1].elliptical_sersic.centre.centre_0 = model_mapper.UniformPrior(1.0, 3.0)
            self.lens_galaxies[1].elliptical_sersic.centre.centre_1 = model_mapper.UniformPrior(1.0, 3.0)

    def modify_mask_function(img):
        return msk.Mask.circular(img.shape_arc_seconds, pixel_scale=img.pixel_scale, radius_mask=5.)

    phase1 = LensPlanex2GalPhase(lens_galaxies=[gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic),
                                                gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic)],
                                 mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                 phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.Pipeline(pipeline_name, phase1)

if __name__ == "__main__":
    test_lens_x2_gal_pipeline()