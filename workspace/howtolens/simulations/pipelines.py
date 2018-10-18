from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import imaging_plotters
import os

def pipeline_lens_and_soure_image():

    psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.1, pixel_scale=0.1)

    image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(130, 130), pixel_scale=0.1, psf_shape=(11, 11))

    lens_galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6))
    source_galaxy = g.Galaxy(light=lp.SphericalExponential(centre=(0.0, 0.0), intensity=0.2, effective_radius=0.2))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=image_plane_grids)

    image_simulated = im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                                   exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

    path = "{}/../3_pipelines/".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
    im.output_imaging_to_fits(image=image_simulated, image_path=path+'/data/1_lens_and_source/image.fits',
                                                     noise_map_path=path+'/data/1_lens_and_source/noise_map.fits',
                                                     psf_path=path+'/data/1_lens_and_source/psf.fits',
                              overwrite=True)

def pipeline_x2_lens_galaxies_image():

    psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(180, 180), pixel_scale=0.05, psf_shape=(11, 11))

    lens_galaxy_0 = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, -1.0), axis_ratio=0.8, phi=55.0, intensity=0.1,
                                                       effective_radius=0.8, sersic_index=2.5),
                             mass=mp.EllipticalIsothermal( centre=(1.0, 0.0), axis_ratio=0.7, phi=45.0, einstein_radius=1.0))
    lens_galaxy_1 = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 1.0), axis_ratio=0.8, phi=100.0, intensity=0.1,
                                                       effective_radius=0.6, sersic_index=3.0),
                             mass=mp.EllipticalIsothermal(centre=(-1.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=0.8))
    source_galaxy = g.Galaxy(light=lp.SphericalExponential(centre=(0.05, 0.15), intensity=0.2, effective_radius=0.5))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy_0, lens_galaxy_1],
                                                 source_galaxies=[source_galaxy], image_plane_grids=image_plane_grids)

    image_simulated = im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.05,
                                                   exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

    path = "{}/../3_pipelines/".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
    im.output_imaging_to_fits(image=image_simulated, image_path=path+'/data/2_x2_lens_galaxies/image.fits',
                                                     noise_map_path=path+'/data/2_x2_lens_galaxies/noise_map.fits',
                                                     psf_path=path+'/data/2_x2_lens_galaxies/psf.fits', overwrite=True)

    #imaging_plotters.plot_image_subplot(image=image_simulated)

def pipeline_complex_source_image():

    psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(180, 180), pixel_scale=0.05, psf_shape=(11, 11))

    lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal( centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0,
                                                         einstein_radius=1.6))
    source_galaxy_0 = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=90.0, intensity=0.2,
                                                         effective_radius=1.0, sersic_index=1.5))
    source_galaxy_1 = g.Galaxy(light=lp.EllipticalSersic(centre=(-0.25, 0.25), axis_ratio=0.7, phi=45.0, intensity=0.1,
                                                         effective_radius=0.2, sersic_index=3.0))
    source_galaxy_2 = g.Galaxy(light=lp.EllipticalSersic(centre=(0.45, -0.35), axis_ratio=0.6, phi=90.0, intensity=0.03,
                                                         effective_radius=0.3, sersic_index=3.5))
    source_galaxy_3 = g.Galaxy(light=lp.EllipticalSersic(centre=(-0.05, -0.0), axis_ratio=0.9, phi=140.0, intensity=0.03,
                                                         effective_radius=0.1, sersic_index=4.0))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy],
                                                 source_galaxies=[source_galaxy_0, source_galaxy_1,
                                                                  source_galaxy_2, source_galaxy_3],
                                                 image_plane_grids=image_plane_grids)

    image_simulated = im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.05,
                                                   exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

    path = "{}/../3_pipelines/".format(os.path.dirname(os.path.realpath(__file__))) # Setup path so we can output the simulated data.
    im.output_imaging_to_fits(image=image_simulated, image_path=path+'/data/3_complex_source/image.fits',
                                                     noise_map_path=path+'/data/3_complex_source/noise_map.fits',
                                                     psf_path=path+'/data/3_complex_source/psf.fits', overwrite=True)