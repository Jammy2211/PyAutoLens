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

def test_lens_x2_gal_hyper_pipeline():

    pipeline_name = "lens_x2_gal_hyper"
    data_name = '/lens_x2_gal_hyper'

    try:
        shutil.rmtree(dirpath+'/data'+data_name)
    except FileNotFoundError:
        pass

    bulge_0 = lp.EllipticalSersic(centre=(-2.0, -2.0), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                 effective_radius=1.0, sersic_index=4.0)
    disk_0 = lp.EllipticalSersic(centre=(-2.0, -2.0), axis_ratio=0.6, phi=90.0, intensity=1.0,
                                 effective_radius=2.5, sersic_index=1.0)
    bulge_1 = lp.EllipticalSersic(centre=(2.0, 2.0), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                 effective_radius=1.0, sersic_index=4.0)
    disk_1 = lp.EllipticalSersic(centre=(2.0, 2.0), axis_ratio=0.6, phi=90.0, intensity=1.0,
                                 effective_radius=2.5, sersic_index=1.0)

    lens_galaxy_0 = galaxy.Galaxy(bulge=bulge_0, disk=disk_0)
    lens_galaxy_1 = galaxy.Galaxy(bulge=bulge_1, disk=disk_1)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy_0, lens_galaxy_1],
                                     source_galaxies=[])

    conf.instance.output_path = output_path

    # try:
    #     shutil.rmtree(output_path + pipeline_name)
    # except FileNotFoundError:
    #     pass

    pipeline = make_lens_x2_gal_hyper_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)

def make_lens_x2_gal_hyper_pipeline(pipeline_name):

    from autolens.pipeline import phase as ph
    from autolens.analysis import galaxy_prior as gp
    from autolens.imaging import mask as msk
    from autolens.profiles import light_profiles

    def modify_mask_function(img):
        return msk.Mask.circular(img.shape_arc_seconds, pixel_scale=img.pixel_scale, radius_mask=5.)

    class LensPlaneGalaxy0Phase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0].elliptical_sersic.centre.centre_0 = model_mapper.UniformPrior(-3.0, -1.0)
            self.lens_galaxies[0].elliptical_sersic.centre.centre_1 = model_mapper.UniformPrior(-3.0, -1.0)

    phase1 = LensPlaneGalaxy0Phase(lens_galaxies=[gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic)],
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                   phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class LensPlaneGalaxy1Phase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0].elliptical_sersic = previous_results[0].constant.lens_galaxies[0].elliptical_sersic
            self.lens_galaxies[1].elliptical_sersic.centre.centre_0 = model_mapper.UniformPrior(1.0, 3.0)
            self.lens_galaxies[1].elliptical_sersic.centre.centre_1 = model_mapper.UniformPrior(1.0, 3.0)

    phase2 = LensPlaneGalaxy1Phase(lens_galaxies=[gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic),
                                                  gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic)],
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                   phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    phase2h = ph.LensPlaneHyperOnlyPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase2h".format(pipeline_name))

    class LensPlaneBothGalaxyPhase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            for i in range(len(self.lens_galaxies)):
                self.lens_galaxies[i].elliptical_sersic = previous_results[-1].variable.lens_galaxies[i].elliptical_sersic
                self.lens_galaxies[i].hyper_galaxy = previous_results[-1].hyper.constant.lens_galaxies[i].hyper_galaxy

    phase3 = LensPlaneBothGalaxyPhase(lens_galaxies=[gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic),
                                                     gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic)],
                                      mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                      phase_name="{}/phase3".format(pipeline_name))

    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return pl.Pipeline(pipeline_name, phase1, phase2, phase2h, phase3)

if __name__ == "__main__":
    test_lens_x2_gal_hyper_pipeline()