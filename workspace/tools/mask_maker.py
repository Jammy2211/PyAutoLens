from autolens.data import ccd
from autolens.data.array import mask as msk
from autolens.data.plotters import data_plotters

import os

# This tool allows one to mask a bespoke mask for a given image of a strong lens, which can then be loaded before a
# pipeline is run and passed to that pipeline so as to become the default masked used by a phase (if a mask
# function is not passed to that phase).

# The 'lens name' is the name of the lens in the data folder, e.g. if you run this, code the mask will be output as
# '/workspace/data/example/mask.fits)
lens_name = 'example'

# First, load the CCD imaging data, so that the mask can be plotted over the strong lens image.
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))
image = ccd.load_image(image_path=path+'/data/'+lens_name+'/image.fits', image_hdu=0, pixel_scale=0.03)

# Now, create a mask for this data, using the mask function's we're used to. I'll use a circular-annular mask here,
# but I've commented over options you might want to use (feel free to experiment!)

mask = msk.Mask.circular_annular(shape=image.shape, pixel_scale=image.pixel_scale,
                                 inner_radius_arcsec=0.2, outer_radius_arcsec=3.0, centre=(0.0, 0.0))

# mask = msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale,
#                          radius_arcsec=3.0, centre=(0.0, 0.0))

# mask = msk.Mask.elliptical(shape=image.shape, pixel_scale=image.pixel_scale,
#                            major_axis_radius_arcsec=3.0, axis_ratio=0.8, phi=45.0, centre=(0.0, 0.0))

# mask = msk.Mask.elliptical_annular(shape=image.shape, pixel_scale=image.pixel_scale,
#                                    inner_major_axis_radius_arcsec=1.0, inner_axis_ratio=0.8, inner_phi=45.0,
#                                    outer_major_axis_radius_arcsec=1.0, outer_axis_ratio=0.6, outer_phi=60.0,
#                                    centre=(0.0, 0.0))

# Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
data_plotters.plot_image(image=image, mask=mask)

# Now we're happy with the mask, lets output it to the data folder the lens, so that we can load it from a .fits
# file in our pipelines!
msk.output_mask_to_fits(mask=mask, mask_path=path+'/data/'+lens_name+'/mask.fits')