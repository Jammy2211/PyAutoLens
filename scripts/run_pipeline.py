from autolens.pipeline import source_pix_pipeline
from autolens.imaging import image as im
from autolens.imaging import scaled_array
from autolens import conf

import numpy as np
import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
output_path = '/gpfs/data/pdtw24/Lens/'

def load_image(data_name, pixel_scale):

    data_dir = "{}/../data/{}".format(dirpath, data_name)

    data = scaled_array.ScaledArray.from_fits_with_scale(file_path=data_dir + '/image', hdu=0, pixel_scale=pixel_scale)
    noise = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/noise', hdu=0)
    psf = im.PSF.from_fits(file_path=data_dir + '/psf', hdu=0)

    return im.Image(array=data, pixel_scale=pixel_scale, psf=psf, noise=noise)

image = load_image(data_name='source_sub3', pixel_scale=0.05)

conf.instance.output_path = output_path
pipeline = source_pix_pipeline.make()
pipeline.run(image)