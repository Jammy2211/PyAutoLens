from autolens.imaging import imaging_util
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.lensing import ray_tracing
from autolens.lensing import galaxy as g
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.visualize import image_plotters
from autolens.visualize import array_plotters
import os

# In this example, we'll simulate a lensed source galaxy and output the images (as .fits). This _image is used to
# demonstrate lens modeling in example/phase/2_phase.py.

# First, lets setup the PSF we are going to blur our simulated _image with, using a Gaussian profile on an 11x11 grid.
psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.75)
array_plotters.plot_psf(psf=psf)

# We need to set up the grids of Cartesian coordinates we will use to perform ray-tracing. The function below
# sets these grids up using the shape and pixel-scale of the _image we will ultimately simulate. The PSF shape is
# required to ensure that edge-effects do not impact PSF blurring later in the simulation.
imaging_grids = mask.ImagingGrids.unmasked_grids_for_simulation(shape=(100, 100), pixel_scale=0.1, psf_shape=psf.shape)

# Now lets make our lens and source galaxies, using the 'galaxy' module (imported as 'g'), 'light_profiles' module
# (imported as 'lp') and 'mass profiles' module (imported as 'mp') to setup the lens and source galaxies.

# For the lens galaxy, we'll use a singular isothermal ellipsoid (SIE) mass profile.
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=40.0, einstein_radius=1.8))

# And for the source galaxy an elliptical Exponential profile.
source_galaxy = g.Galaxy(light=lp.EllipticalExponential(centre=(0.01, 0.01), axis_ratio=0.9, phi=90.0, intensity=0.5,
                                                        effective_radius=0.3))

# Next, we pass these galaxies into the 'ray_tracing' module, here using a 'TracerImageSourcePlanes' which represents
# ray-tracer which has both an _image and source plane.

# Using the lens galaxy's mass profile(s), this tracer automatically computes the deflection angles of light on the
# imaging grids and traces their coordinates to the source-plane
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grids=imaging_grids)

# For this example, we'll extract the _image-plane _image of this ray-tracing geometry. We actually need a special _image
# specifically for simulations, which ensures that edge-effects do not degrade the PSF convolution.
array_plotters.plot_image(image=tracer.image_plane_image_for_simulation)

# To simulate the _image, we pass this _image to the imaging module's simulate function. We add various effects,
# including PSF blurring, the background sky and noise.
image_simulated = image.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                                exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Now we can visualize the _image, noise-map and psf. Note we're using an object plotter (rather than the array
# plotters that we used above), which automatically plots all of the simulated _image data.
# Now we're using an _image with a defined pixel scale, the x and y axis of these figures are in arc-seconds
image_plotters.plot_image(image=image_simulated)

# Finally, lets output these files to.fits so that we can fit them in the phase and pipeline examples
path = "{}".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
imaging_util.numpy_array_to_fits(array=image_simulated, path=path+'/../data/1_basic/_image.fits')
imaging_util.numpy_array_to_fits(array=image_simulated.noise_map, path=path + '/../data/1_basic/noise_map.fits')
imaging_util.numpy_array_to_fits(array=psf, path=path+'/../data/1_basic/psf.fits')