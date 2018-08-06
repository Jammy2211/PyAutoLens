import numpy as np
import os
import sys
from matplotlib import pyplot
sys.path.append("../")

from autolens.imaging import imaging, simulate
from astropy.io import fits

def numpy_array_to_fits(array, path, filename):

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array, new_hdr)
    hdu.writeto(path + filename + '.fits')

data_path = "{}/../weighted_data/".format(os.path.dirname(os.path.realpath(__file__)))

pixel_scale = 0.05
dimensions = (256, 256)

psf_dim = (15, 15)

kernel = np.zeros(psf_dim)
sigma = 0.05

for y in range(psf_dim[0]):
    for x in range(psf_dim[1]):

        y0 = (y-((psf_dim[0]/2.0)-0.5))*0.05
        x0 = (x-((psf_dim[1]/2.0)-0.5))*0.05

        kernel[y,x] = np.exp(-(((x0**2)/(2*sigma**2)) + ((y0**2)/(2*sigma**2))))

# psf = imaging.PSF.from_fits(path=data_path+'slacs/', filename='slacs_1_post.fits', hdu=3, pixel_scale=pixel_scale)
# psf.trim(new_dimensions=(15,15))

psf = imaging.PSF(data=kernel, pixel_scale=pixel_scale, renormalize=True)


exposure_time = imaging.ExposureTime.from_one_value(exposure_time=5e3, pixel_dimensions=dimensions,
                                                    pixel_scale=pixel_scale)

background_noise = imaging.NoiseBackground.from_one_value(background_noise=0.000005, pixel_dimensions=dimensions,
                                                          pixel_scale=pixel_scale)

image = simulate.SimulateImage.from_fits(path=data_path, filename='SimLens.fits', hdu=0, pixel_scale=pixel_scale,
                                         exposure_time=exposure_time,
                                         sim_optics=simulate.SimulateOptics(psf=psf),
                                         sim_poisson_noise=simulate.SimulatePoissonNoise(),
                                         sim_background_noise=simulate.SimulateBackgroundNoise(background_noise_sigma=0.005))

noise = imaging.estimate_noise_from_image(image.data, image.exposure_time.data, background_noise.data)
signal_to_noise_ratio = np.divide(image.data, noise)

print(np.max(image.data_original), np.min(image.data_original))
print(np.max(image.data), np.min(image.data))
print(np.max(noise), np.min(noise))
print(np.max(signal_to_noise_ratio))

# pyplot.imshow(psf.weighted_data)
# pyplot.show()
image.plot()

numpy_array_to_fits(image.data, path=data_path, filename='NanSim')

imaging.output_for_fortran(path=data_path, array=image, image_name='NanSim')
imaging.output_for_fortran(path=data_path, array=psf, image_name='NanSim')
imaging.output_for_fortran(path=data_path, array=imaging.Noise(noise, pixel_scale), image_name='NanSim')