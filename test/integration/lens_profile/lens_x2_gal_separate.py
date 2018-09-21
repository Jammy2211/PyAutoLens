from autolens.pipeline import pipeline as pl
from autolens.pipeline import phase as ph
from autolens.profiles import light_profiles as lp
from autolens.lensing import galaxy
from autolens.lensing import galaxy_model as gp
from autolens.imaging import mask as msk
from autolens.autofit import non_linear as nl
from autolens.autofit import model_mapper as mm
from test.integration import tools
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '/gpfs/data/pdtw24/Lens/int/lens_profile/'


def test_lens_x2_gal_separate_pipeline():
    pipeline_name = "l2g_sep"
    data_name = '/l2g_sep'

    tools.reset_paths(data_name, pipeline_name, output_path)

    sersic_0 = lp.EllipticalSersic(centre=(-1.0, -1.0), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                   effective_radius=1.3, sersic_index=3.0)

    sersic_1 = lp.EllipticalSersic(centre=(1.0, 1.0), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                   effective_radius=1.3, sersic_index=3.0)

    lens_galaxy_0 = galaxy.Galaxy(light_profile=sersic_0)
    lens_galaxy_1 = galaxy.Galaxy(light_profile=sersic_1)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy_0, lens_galaxy_1],
                                     source_galaxies=[], target_signal_to_noise=50.0)

    pipeline = make_lens_x2_gal_separate_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)


def make_lens_x2_gal_separate_pipeline(pipeline_name):
    def modify_mask_function(img):
        return msk.Mask.circular(shape=img.shape, pixel_scale=img.pixel_scale, radius_mask_arcsec=5.)

    class LensPlaneGalaxy0Phase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0].elliptical_sersic.centre_0 = -1.0
            self.lens_galaxies[0].elliptical_sersic.centre_1 = -1.0

    phase1 = LensPlaneGalaxy0Phase(lens_galaxies=[gp.GalaxyModel(elliptical_sersic=lp.EllipticalSersic)],
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                   phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class LensPlaneGalaxy1Phase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0] = previous_results[-1].constant.lens_galaxies[0]
            self.lens_galaxies[1].elliptical_sersic.centre_0 = mm.UniformPrior(0.0, 2.0)
            self.lens_galaxies[1].elliptical_sersic.centre_1 = mm.UniformPrior(0.0, 2.0)

    phase2 = LensPlaneGalaxy1Phase(lens_galaxies=[gp.GalaxyModel(elliptical_sersic=lp.EllipticalSersic),
                                                  gp.GalaxyModel(elliptical_sersic=lp.EllipticalSersic)],
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                   phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    class LensPlaneBothGalaxyPhase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0] = previous_results[0].variable.lens_galaxies[0]
            self.lens_galaxies[1] = previous_results[1].variable.lens_galaxies[1]

    phase3 = LensPlaneBothGalaxyPhase(lens_galaxies=[gp.GalaxyModel(elliptical_sersic=lp.EllipticalSersic),
                                                     gp.GalaxyModel(elliptical_sersic=lp.EllipticalSersic)],
                                      mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                      phase_name="{}/phase3".format(pipeline_name))

    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(pipeline_name, phase1, phase2, phase3)


if __name__ == "__main__":
    test_lens_x2_gal_separate_pipeline()
