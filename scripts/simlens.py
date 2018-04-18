import numpy as np
import os
import sys
sys.path.append("../")

from auto_lens.imaging import imaging, simulate

data_path = "{}/../data/".format(os.path.dirname(os.path.realpath(__file__)))

pixel_scale = 0.05

psf = imaging.PSF.from_fits(path=data_path+'slacs/', filename='slacs_1_post.fits', hdu=3, pixel_scale=pixel_scale)
psf.trim(new_dimensions=(15,15))

exposure_time = imaging.ExposureTimeMap.from_single_exposure_time(exposure_time=1e7, pixel_dimensions=(602, 602),
                                                                  pixel_scale=pixel_scale)

image = simulate.SimulateImage.from_fits(path=data_path, filename='SimLens.fits', hdu=0, pixel_scale=pixel_scale,
                                         psf=psf, exposure_time_map=exposure_time, sky_level=0.001)

print(np.max(image.data_original), np.min(image.data_original))
print(np.max(image.data), np.min(image.data))
print(np.max(image.noise), np.min(image.noise))

snr = np.divide(image.data, image.noise)

print(np.max(snr))
image.plot()

imaging.output_for_fortran(path=data_path, array=image, image_name='NanSim')
imaging.output_for_fortran(path=data_path, array=psf, image_name='NanSim')
imaging.output_for_fortran(path=data_path, array=imaging.Noise(image.noise, pixel_scale=pixel_scale), image_name='NanSim')

