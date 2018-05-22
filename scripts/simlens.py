import numpy as np
import os
import sys

sys.path.append("../")

from auto_lens.imaging import imaging, simulate
from astropy.io import fits


def numpy_array_to_fits(array, path, filename):
    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array, new_hdr)
    hdu.writeto(path + filename + '.fits')


data_path = "{}/../data/".format(os.path.dirname(os.path.realpath(__file__)))

pixel_scale = 0.05
dimensions = (256, 256)

psf = imaging.PSF.from_fits(path=data_path + 'slacs/', filename='slacs_1_post.fits', hdu=3, pixel_scale=pixel_scale)
psf.trim(new_dimensions=(15, 15))

exposure_time = imaging.ExposureTime.from_one_value(exposure_time=5e8, pixel_dimensions=dimensions,
                                                    pixel_scale=pixel_scale)

background_noise = imaging.NoiseBackground.from_one_value(background_noise=0.000005, pixel_dimensions=dimensions,
                                                          pixel_scale=pixel_scale)

image = simulate.SimulateImage.from_fits(path=data_path, filename='SimLens.fits', hdu=0, pixel_scale=pixel_scale,
                                         exposure_time=exposure_time,
                                         sim_optics=simulate.SimulateOptics(psf=psf))

print(np.max(image.data_original), np.min(image.data_original))
print(np.max(image.data), np.min(image.data))
# print(np.max(image.noise), np.min(image.noise))
# print(np.max(image.signal_to_noise_ratio))

image.plot()

numpy_array_to_fits(image.data, path=data_path, filename='NanSim')

imaging.output_for_fortran(path=data_path, array=image, image_name='NanSim')
imaging.output_for_fortran(path=data_path, array=psf, image_name='NanSim')
imaging.output_for_fortran(path=data_path, array=imaging.Noise(image.noise, pixel_scale), image_name='NanSim')
