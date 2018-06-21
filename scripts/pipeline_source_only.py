from auto_lens.imaging import image as im
from auto_lens.imaging import mask as msk
from auto_lens.imaging import scaled_array
from auto_lens.analysis import pipeline
import os

lens_name = 'source_only_pipeline'
data_dir = "../data/" + lens_name.format(os.path.dirname(os.path.realpath(__file__)))

image = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/image', hdu=0, pixel_scale=0.1)
noise = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/noise', hdu=0, pixel_scale=0.1)
exposure_time = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/exposure_time', hdu=0,
                                                   pixel_scale=0.1)
psf = im.PSF.from_fits(file_path=data_dir + '/psf', hdu=0, pixel_scale=0.1)

data = im.Image(array=image, effective_exposure_time=exposure_time, pixel_scale=0.1, psf=psf,
                background_noise=noise, poisson_noise=noise)

mask = msk.Mask.circular(shape_arc_seconds=data.shape_arc_seconds, pixel_scale=data.pixel_scale, radius_mask=2.0)

results = pipeline.source_only_pipeline(image, mask)

print(results)
