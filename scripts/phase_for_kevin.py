from autolens.autofit import non_linear
from autolens.autofit import model_mapper
from autolens.pipeline import phase as ph
from autolens.lensing import galaxy_model as gm
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import imaging_plotters
from autolens.plotting import fitting_plotters
import os
import numpy as np

path = "{}".format(os.path.dirname(os.path.realpath(__file__)))
image = im.load_imaging_from_fits(image_path=path + '/../data/kevin/v03_drz_final5-ac.fits', image_hdu=1,
                                  noise_map_path=path+'/../data/kevin/v22_drz_final5-ac-WHT.fits', noise_map_hdu=1,
                                  psf_path=path + '/../data/kevin/v22_psf.fits', psf_hdu=0,
                                  pixel_scale=0.05, noise_map_is_weight_map=True)

image = image.trim_image_and_noise_around_region(x0=1042, x1=1342, y0=900, y1=1200)
imaging_plotters.plot_image_individuals(image=image, plot_image=True, plot_noise_map=False)


class CustomPhase(ph.LensSourcePlanePhase):

    def pass_priors(self, previous_results):
        self.lens_galaxies[0].mass.centre_0 = 0.1
        self.lens_galaxies[0].mass.centre_1 = 0.1

        self.lens_galaxies[0].mass.einstein_radius = model_mapper.UniformPrior(lower_limit=1.4, upper_limit=1.8)


def mask_function(img):
    return mask.Mask.annular(img.shape, pixel_scale=img.pixel_scale, inner_radius_arcsec=0.4,
                             outer_radius_arcsec=2.8)

phase = CustomPhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic, mass=mp.EllipticalIsothermal)],
                    source_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                    optimizer_class=non_linear.MultiNest, phase_name='phase0')

results = phase.run(image)
fitting_plotters.plot_fitting(fit=results.fit)

# If a phase has already begun running at its specified path, the analysis resumes from where it was terminated.
# The folder must be manually deleted to start from scratch. So, if you run this script again, you'll notice the
# results appear immediately!
results_1 = phase.run(image)
print(results_1) # NOTE - this isn't working yet, need to sort out.
fitting_plotters.plot_fitting(fit=results_1.fit)
