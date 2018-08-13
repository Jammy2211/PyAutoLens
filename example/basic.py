from autolens.imaging import scaled_array
from autolens.imaging import image as im

pixel_scale = 0.05

data = scaled_array.ScaledArray.from_fits_with_scale(file_path='basic/image', hdu=0, pixel_scale=pixel_scale)
noise = scaled_array.ScaledArray.from_fits_with_scale(file_path='basic/noise', hdu=0, pixel_scale=pixel_scale)
psf = im.PSF.from_fits(file_path='basic/psf', hdu=0)
image = im.Image(array=data, pixel_scale=pixel_scale, psf=psf, noise=noise)
