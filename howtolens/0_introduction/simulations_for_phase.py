from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import imaging_plotters
import os

# These are the simulated images we fit in tutorial 5_phase

psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.75)

image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(130, 130), pixel_scale=0.1, psf_shape=(11, 11))

lens_galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(0.1, 0.1), einstein_radius=1.6))
source_galaxy = g.Galaxy(light=lp.SphericalExponential(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=image_plane_grids)

image_simulated = im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                               exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)
imaging_plotters.plot_image(image=image_simulated)

path = "{}".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
im.output_imaging_to_fits(image=image_simulated, image_path=path+'/data/phase_simple_image.fits',
                                                 noise_map_path=path+'/data/phase_simple_noise_map.fits',
                                                 psf_path=path+'/data/phase_simple_psf.fits',
                          overwrite=True)