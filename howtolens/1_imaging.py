from autolens.imaging import image
from autolens.imaging import mask
from autolens.plotting import imaging_plotters
from autolens.plotting import array_plotters

import os

# AutoLens

# Setup the path of the analysis so we can load the example data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

# Load an _image from the 'data/1_basic' folder.
image = image.load_data(image_path=path + '/../data/2_imaging/image.fits', image_hdu=0,
                        noise_map_path=path+'/../data/2_imaging/noise_map.fits', noise_map_hdu=0,
                        psf_path=path + '/../data/2_imaging/psf.fits', psf_hdu=0,
                        pixel_scale=0.1)

imaging_plotters.plot_image(image=image)
imaging_plotters.plot_image_individuals(image=image)

ma = mask.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_mask_arcsec=3.0)

