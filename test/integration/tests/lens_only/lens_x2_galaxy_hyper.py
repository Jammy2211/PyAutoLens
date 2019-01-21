import os

from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.data import ccd
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration import tools

test_type = 'lens_only'
test_name = "lens_x2_galaxy_hyper"

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path+'output/'+test_type
config_path = path+'config'
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def test_pipeline():

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

    tools.reset_paths(test_name=test_name, output_path=output_path)
    tools.simulate_integration_image(test_name=test_name, pixel_scale=0.1, lens_galaxies=[lens_galaxy_0, lens_galaxy_1],
                                     source_galaxies=[], target_signal_to_noise=30.0)

    ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/' + test_name + '/image.fits',
                                        psf_path=path + '/data/' + test_name + '/psf.fits',
                                        noise_map_path=path + '/data/' + test_name + '/noise_map.fits',
                                        pixel_scale=0.1)

    pipeline = make_pipeline(test_name=test_name)
    pipeline.run(data=ccd_data)

def make_pipeline(test_name):
    def modify_mask_function(img):
        return msk.Mask.circular(shape=img.shape, pixel_scale=img.pixel_scale, radius_arcsec=5.)

    class LensPlaneGalaxy0Phase(ph.LensPlanePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0].sersic.centre_0 = -1.0
            self.lens_galaxies[0].sersic.centre_1 = -1.0

    phase1 = LensPlaneGalaxy0Phase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                                   mask_function=modify_mask_function, optimizer_class=nl.MultiNest,
                                   phase_name="{}/phase1".format(test_name))

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
                                   phase_name="{}/phase2".format(test_name))

    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    phase2h = ph.LensLightHyperOnlyPhase(optimizer_class=nl.MultiNest, phase_name="{}/phase2h".format(test_name),
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
                                      phase_name="{}/phase3".format(test_name))

    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(test_name, phase1, phase2, phase2h, phase3)


if __name__ == "__main__":
    test_pipeline()
