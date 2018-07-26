import sys
sys.path.append("../")

from src.imaging import image as im
from src.imaging import mask as msk
from src.imaging import scaled_array
from src.analysis import pipeline
import os

# Load up the data
data_name = 'lens_sersic'
paths = pipeline.PipelinePaths(data_name=data_name)

data = scaled_array.ScaledArray.from_fits(file_path=paths.data_path + '/image', hdu=0, pixel_scale=0.08)
noise = scaled_array.ScaledArray.from_fits(file_path=paths.data_path + '/noise', hdu=0, pixel_scale=0.08)
exposure_time = scaled_array.ScaledArray.from_fits(file_path=paths.data_path + '/exposure_time', hdu=0, pixel_scale=0.08)
psf = im.PSF.from_fits(file_path=paths.data_path + '/psf', hdu=0, pixel_scale=0.08)

image = im.Image(array=data, effective_exposure_time=exposure_time, pixel_scale=0.08, psf=psf, background_noise=noise,
                 poisson_noise=noise)

mask = msk.Mask.circular(shape_arc_seconds=image.shape_arc_seconds, pixel_scale=data.pixel_scale, radius_mask=2.4)

# Run the primary pipeline
results = pipeline.profiles_pipeline(paths, image, mask)

# Print results
print(results)
