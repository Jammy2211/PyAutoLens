from autolens.data import ccd
from autolens.data.array import grids
from autolens.lens import ray_tracing
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.lens.plotters import ray_tracing_plotters
from autolens.data.plotters import ccd_plotters

import os

# This tool allows one to make simulated data-sets of strong lenses using a multi-plane ray-tracer, such that all
# galaxies down the line-of-sight are included in the ray-tracing calculation based on their redshifts.

# The 'lens name' is the name of the lens in the data folder, e.g:

# The image will be output as '/workspace/data/example1/image.fits'.
# The noise-map will be output as '/workspace/data/example1/noise_map.fits'.
# The psf will be output as '/workspace/data/example1/psf.fits'.

# (these files are already in the workspace and are remade running this script)
lens_name = 'multi_plane'
pixel_scale = 0.05

# Simulate a simple Gaussian PSF for the image.
psf = ccd.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=pixel_scale)

# Setup the image-plane grid stack of the CCD array which will be used for generating the image-plane image of the
# simulated strong lens.
image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(shape=(400, 400), pixel_scale=pixel_scale,
                                                                   psf_shape=psf.shape, sub_grid_size=1)

# Setup the lens galaxy's light mass (SIE) and source galaxy light (elliptical Sersic) for this simulated lens.
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0),
                       shear=mp.ExternalShear(magnitude=0.05, phi=90.0),
                       redshift=0.5)

source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=60.0,
                                                   intensity=1.0, effective_radius=1.0, sersic_index=2.5),
                         redshift=1.0)

# Setup our line-of-sight (los) galaxies using Spherical Sersic profiles for their light and Singular
# Isothermal Sphere (SIS) profiles. We'll use 3 galaxies, but you can add more if desired.
los_0 = g.Galaxy(light=lp.SphericalSersic(centre=(4.0, 4.0), intensity=0.30, effective_radius=0.3, sersic_index=2.0),
                mass=mp.SphericalIsothermal(centre=(4.0, 4.0), einstein_radius=0.02),
                redshift=0.25)
los_1 = g.Galaxy(light=lp.SphericalSersic(centre=(3.6, -5.3), intensity=0.20, effective_radius=0.6, sersic_index=1.5),
                 mass=mp.SphericalIsothermal(centre=(3.6, -5.3), einstein_radius=0.04),
                redshift=0.75)
los_2 = g.Galaxy(light=lp.SphericalSersic(centre=(-3.1, -2.4), intensity=0.35, effective_radius=0.4, sersic_index=2.5),
                 mass=mp.SphericalIsothermal(centre=(-3.1, -2.4), einstein_radius=0.03),
                redshift=1.25)

# Use these galaxies to setup a multi-plane tracer, which will generate the image-plane image for the simulated CCD
# data. This tracer orders galaxies by redshift and performs ray-tracing based on their line-of-sight redshifts.
tracer = ray_tracing.TracerMultiPlanes(galaxies=[lens_galaxy, source_galaxy, los_0, los_1, los_2],
                                       image_plane_grid_stack=image_plane_grid_stack)

# Lets look at the tracer's image-plane image - this is the image we'll be simulating.
ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer)

# Now lets simulate the CCD data, remembering that we use a special image-plane image which ensures edge-effects don't
# degrade our modeling of the telescope optics (e.g. the PSF convolution).
simulated_ccd = ccd.CCDData.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=pixel_scale,
                                     exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Lets plot the simulated CCD data before we output it to files.
ccd_plotters.plot_ccd_subplot(ccd_data=simulated_ccd)

# Now, lets output this simulated ccd-data to the data folder (we'll model it in the workspace's example multi-plane
# pipeline).
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Check a folder of the lens_name exists in the data folder for the images to be output. If it doesn't make it.
if not os.path.exists(path+'/data/example/'+lens_name):
    os.makedirs(path+'/data/example/'+lens_name)

image = ccd.output_ccd_data_to_fits(ccd_data=simulated_ccd,
                                    image_path=path+'/data/example/'+lens_name+'/image.fits',
                                    psf_path=path + '/data/example/' + lens_name + '/psf.fits',
                                    noise_map_path=path + '/data/example/' + lens_name + '/noise_map.fits',
                                    overwrite=True)
