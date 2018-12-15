from astropy.io import fits
import os

from autolens.data.imaging import image as im
from autolens.data.imaging.plotters import imaging_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp


def simulate_image():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grids = grids.DataGridStack.grids_for_simulation(shape=(100, 100), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0,
                                                    sersic_index=2.0),
                           mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0,
                                                      sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=[image_plane_grids])

    return im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_image_in_counts():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grids = grids.DataGridStack.grids_for_simulation(shape=(100, 100), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=1000*0.3, effective_radius=1.0,
                                                    sersic_index=2.0),
                           mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=1000*0.2, effective_radius=1.0,
                                                      sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=[image_plane_grids])

    return im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_image_large_stamp():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grids = grids.DataGridStack.grids_for_simulation(shape=(500, 500), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0,
                                                    sersic_index=2.0),
                           mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0,
                                                      sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=[image_plane_grids])

    return im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_image_small_stamp():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grids = grids.DataGridStack.grids_for_simulation(shape=(50, 50), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0,
                                                    sersic_index=2.0),
                           mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0,
                                                      sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=[image_plane_grids])

    return im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_image_offset_centre():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grids = grids.DataGridStack.grids_for_simulation(shape=(100, 100), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(1.0, 1.0), intensity=0.3, effective_radius=1.0,
                                                    sersic_index=2.0),
                           mass=mp.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(1.0, 1.0), intensity=0.2, effective_radius=1.0,
                                                      sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=[image_plane_grids])

    return im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_image_with_large_psf():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(101, 101), sigma=0.05, pixel_scale=0.1)

    image_plane_grids = grids.DataGridStack.grids_for_simulation(shape=(100, 100), pixel_scale=0.1, psf_shape=(101, 101))

    lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0,
                                                    sersic_index=2.0),
                           mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0,
                                                      sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=[image_plane_grids])

    return im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))

def simulate_image_with_offset_centre_psf():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1, centre=(0.1, 0.1))

    image_plane_grids = grids.DataGridStack.grids_for_simulation(shape=(100, 100), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0,
                                                    sersic_index=2.0),
                           mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0,
                                                      sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=[image_plane_grids])

    return im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))


def simulate_all_images(plot_images=False):

    image = simulate_image()
    im.output_imaging_to_fits(image=image, image_path=path + '/datas/datas/datas.fits',
                              noise_map_path=path + '/datas/datas/noise_maps.fits',
                              psf_path=path + '/datas/datas/psf.fits', overwrite=True)

    new_hdul = fits.HDUList()
    new_hdul.append(fits.ImageHDU(image))
    new_hdul.append(fits.ImageHDU(image.noise_map))
    new_hdul.append(fits.ImageHDU(image.psf))
    new_hdul.append(fits.ImageHDU(image.exposure_time_map))

    if os.path.exists(path + 'datas/datas/multiple_hdus.fits'):
        os.remove(path+ 'datas/datas/multiple_hdus.fits')
    new_hdul.writeto(path + '/datas/datas/multiple_hdus.fits')

    image_in_counts = simulate_image_in_counts()
    im.output_imaging_to_fits(image=image_in_counts, image_path=path + '/datas/image_in_counts/datas.fits',
                              noise_map_path=path + '/datas/image_in_counts/noise_maps.fits',
                              psf_path=path + '/datas/image_in_counts/psf.fits',
                              exposure_time_map_path=path + '/datas/image_in_counts/exposure_time_map.fits',
                              overwrite=True)

    image_large_stamp = simulate_image_large_stamp()
    im.output_imaging_to_fits(image=image_large_stamp, image_path=path + '/datas/image_large_stamp/datas.fits',
                              noise_map_path=path + '/datas/image_large_stamp/noise_maps.fits',
                              psf_path=path + '/datas/image_large_stamp/psf.fits', overwrite=True)

    image_small_stamp = simulate_image_small_stamp()
    im.output_imaging_to_fits(image=image_small_stamp, image_path=path + '/datas/image_small_stamp/datas.fits',
                              noise_map_path=path + '/datas/image_small_stamp/noise_maps.fits',
                              psf_path=path + '/datas/image_small_stamp/psf.fits', overwrite=True)

    image_offset_centre = simulate_image_offset_centre()
    im.output_imaging_to_fits(image=image_offset_centre, image_path=path + '/datas/image_offset_centre/datas.fits',
                              noise_map_path=path + '/datas/image_offset_centre/noise_maps.fits',
                              psf_path=path + '/datas/image_offset_centre/psf.fits', overwrite=True)

    image_with_large_psf = simulate_image_with_large_psf()
    im.output_imaging_to_fits(image=image_with_large_psf, image_path=path + '/datas/image_with_large_psf/datas.fits',
                              noise_map_path=path + '/datas/image_with_large_psf/noise_maps.fits',
                              psf_path=path + '/datas/image_with_large_psf/psf.fits', overwrite=True)

    image_with_off_centre_psf = simulate_image_with_offset_centre_psf()
    im.output_imaging_to_fits(image=image_with_off_centre_psf,
                              image_path=path + '/datas/image_with_off_centre_psf/datas.fits',
                              noise_map_path=path + '/datas/image_with_off_centre_psf/noise_maps.fits',
                              psf_path=path + '/datas/image_with_off_centre_psf/psf.fits', overwrite=True)


    if plot_images:
        imaging_plotters.plot_image_subplot(image=image)
        imaging_plotters.plot_image_subplot(image=image_in_counts)
        imaging_plotters.plot_image_subplot(image=image_offset_centre)