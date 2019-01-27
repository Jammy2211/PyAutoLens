from autolens.data import ccd
from autolens.data.array import grids
from autolens.lens import ray_tracing
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.lens.plotters import ray_tracing_plotters
from autolens.data.plotters import ccd_plotters

from test.profiling import tools

import os

pixel_scales = [0.2, 0.1, 0.05, 0.03, 0.01]
shapes = [(100, 100), (150, 150), (250, 250), (320, 320), (750, 750)]

def simulate_image_from_galaxies_and_output_to_fits(lens_name, pixel_scale, shape, sub_grid_size, lens_galaxies,
                                                    source_galaxies):

    # Simulate a simple Gaussian PSF for the image.
    psf = ccd.PSF.simulate_as_gaussian(shape=(51, 51), sigma=pixel_scale, pixel_scale=pixel_scale)

    # Setup the image-plane grid stack of the CCD array which will be used for generating the image-plane image of the
    # simulated strong lens. A high-res sub-grid is necessary to ensure we fully resolve the central regions of the
    # lens and source galaxy light.
    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(shape=shape, pixel_scale=pixel_scale,
                                                                       psf_shape=psf.shape, sub_grid_size=sub_grid_size)

    # Use the input galaxies to setup a tracer, which will generate the image-plane image for the simulated CCD data.
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=lens_galaxies, source_galaxies=source_galaxies,
                                                 image_plane_grid_stack=image_plane_grid_stack)

    # Simulate the CCD data, remembering that we use a special image-plane image which ensures edge-effects don't
    # degrade our modeling of the telescope optics (e.g. the PSF convolution).
    simulated_ccd = ccd.CCDData.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=pixel_scale,
                                         exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

    # Now, lets output this simulated ccd-data to the test/profiling/data folder.
    path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))

    # The image_type tag tells us whether we are outputting an image at LSST, Euclid, HST or AO resolution.
    image_type = tools.image_type_from_pixel_scale(pixel_scale=pixel_scale)

    # Check a folder of the lens name and within that of the pixel scale tag exist in the data folder for the images to
    # be output. If it doesn't make it.
    if not os.path.exists(path + '/data/' + lens_name):
        os.makedirs(path + '/data/' + lens_name)

    if not os.path.exists(path + '/data/' + lens_name + '/' + image_type ):
        os.makedirs(path + '/data/' + lens_name + '/' + image_type )

    ccd.output_ccd_data_to_fits(ccd_data=simulated_ccd,
                                image_path=path + '/data/' + lens_name + '/' + image_type + '/image.fits',
                                psf_path=path + '/data/' + lens_name + '/' + image_type + '/psf.fits',
                                noise_map_path=path + '/data/' + lens_name + '/' + image_type + '/noise_map.fits',
                                overwrite=True)

def make_no_lens_source_smooth(sub_grid_size):

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6,
                                                        axis_ratio=0.7, phi=45.0))

    source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=60.0,
                                                       intensity=0.4, effective_radius=0.5, sersic_index=1.0))

    for pixel_scale, shape in zip(pixel_scales, shapes):

        simulate_image_from_galaxies_and_output_to_fits(lens_name='no_lens_source_smooth', shape=shape,
                                                        pixel_scale=pixel_scale, sub_grid_size=sub_grid_size,
                                                        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy])


def make_no_lens_source_cuspy(sub_grid_size):

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6,
                                                        axis_ratio=0.7, phi=45.0))

    source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=60.0,
                                                       intensity=0.1, effective_radius=0.5, sersic_index=3.0))

    for pixel_scale, shape in zip(pixel_scales, shapes):

        simulate_image_from_galaxies_and_output_to_fits(lens_name='no_lens_source_cuspy', shape=shape,
                                                        pixel_scale=pixel_scale, sub_grid_size=sub_grid_size,
                                                        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy])


def make_lens_and_source_smooth(sub_grid_size):

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = g.Galaxy(ight=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0,
                                                 intensity=0.5, effective_radius=0.8, sersic_index=4.0),
                           mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6,
                                                        axis_ratio=0.7, phi=45.0))

    source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=60.0,
                                                       intensity=0.4, effective_radius=0.5, sersic_index=1.0))

    for pixel_scale, shape in zip(pixel_scales, shapes):

        simulate_image_from_galaxies_and_output_to_fits(lens_name='lens_and_source_smooth', shape=shape,
                                                        pixel_scale=pixel_scale, sub_grid_size=sub_grid_size,
                                                        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy])


def make_lens_and_source_cuspy(sub_grid_size):

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = g.Galaxy(ight=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0,
                                                 intensity=0.5, effective_radius=0.8, sersic_index=4.0),
                           mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6,
                                                        axis_ratio=0.7, phi=45.0))

    source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=60.0,
                                                       intensity=0.1, effective_radius=0.5, sersic_index=3.0))

    for pixel_scale, shape in zip(pixel_scales, shapes):

        simulate_image_from_galaxies_and_output_to_fits(lens_name='lens_and_source_cuspy', shape=shape,
                                                        pixel_scale=pixel_scale, sub_grid_size=sub_grid_size,
                                                        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy])