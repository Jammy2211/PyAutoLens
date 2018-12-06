import os

from autofit.core import non_linear as nl
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '{}/../output/lens_only'.format(dirpath)


def test_pipeline():
    pipeline_name = "lens_x2_galaxy_hyper"
    data_name = '/lens_x2_galaxy_hyper'

    tools.reset_paths(data_name, pipeline_name, output_path)

    bulge_0 = lp.EllipticalSersic(centre=(-1.0, -1.0), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                  effective_radius=1.0, sersic_index=4.0)
    disk_0 = lp.EllipticalSersic(centre=(-1.0, -1.0), axis_ratio=0.6, phi=90.0, intensity=20.0,
                                 effective_radius=2.5, sersic_index=1.0)
    bulge_1 = lp.EllipticalSersic(centre=(1.0, 1.0), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                  effective_radius=1.0, sersic_index=4.0)
    disk_1 = lp.EllipticalSersic(centre=(1.0, 1.0), axis_ratio=0.6, phi=90.0, intensity=20.0,
                                 effective_radius=2.5, sersic_index=1.0)

    lens_galaxy_0 = galaxy.Galaxy(bulge=bulge_0, disk=disk_0)
    lens_galaxy_1 = galaxy.Galaxy(bulge=bulge_1, disk=disk_1)

    tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy_0, lens_galaxy_1],
                                     source_galaxies=[], target_signal_to_noise=50.0)

    pipeline = make_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)


def make_pipeline(pipeline_name):
    def modify_mask_function(img):
        return msk.Mask.circular(shape=img.shape, pixel_scale=img.pixel_scale, radius_arcsec=5.)

    class LensPlaneGalaxy0Phase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0].sersic.centre_0 = -1.0
            self.lens_galaxies[0].sersic.centre_1 = -1.0

    phase1 = LensPlaneGalaxy0Phase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                   phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class LensPlaneGalaxy1Phase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0] = previous_results[0].constant.lens_galaxies[0]
            self.lens_galaxies[1].sersic.centre_0 = 1.0
            self.lens_galaxies[1].sersic.centre_1 = 1.0

    phase2 = LensPlaneGalaxy1Phase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                  gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                   phase_name="{}/phase2".format(pipeline_name))

    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    phase2h = ph.LensLightHyperOnlyPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase2h".format(pipeline_name),
                                         mask_function=modify_mask_function)

    class LensPlaneBothGalaxyPhase(ph.LensPlaneHyperPhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0] = previous_results[0].variable.lens_galaxies[0]
            self.lens_galaxies[1] = previous_results[1].variable.lens_galaxies[0]
            self.lens_galaxies[0].hyper_galaxy = previous_results[-1].hyper.constant.lens_galaxies[0].hyper_galaxy
            self.lens_galaxies[1].hyper_galaxy = previous_results[-1].hyper.constant.lens_galaxies[1].hyper_galaxy
            self.lens_galaxies[0].sersic.centre_0 = -1.0
            self.lens_galaxies[0].sersic.centre_1 = -1.0
            self.lens_galaxies[1].sersic.centre_0 = 1.0
            self.lens_galaxies[1].sersic.centre_1 = 1.0

    phase3 = LensPlaneBothGalaxyPhase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic),
                                                     gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                      mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                      phase_name="{}/phase3".format(pipeline_name))

    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(pipeline_name, phase1, phase2, phase2h, phase3)


if __name__ == "__main__":
    test_pipeline()
