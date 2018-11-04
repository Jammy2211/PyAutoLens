import os

from autolens.imaging import image as im
from autolens.plotting import imaging_plotters

# To model data with PyAutoLens, you first need to load it and ensure it is in a format suitable for lens modeling.
# This tutorial takes you though how to do this, alongside a number of PyAutoLens's built in tools that can convert
# data to a suitable format.

# First, lets setup the path to our current working directory. I recommend you use the 'AutoLens/workspace' directory
# and that you place your data in the 'AutoLens/workspace/data' directory.

# (for this tutorial, we'll use the 'AutoLens/workspace/howtolens/preparing_data' directory. The folder 'data' contains
# the example data-sets we'll use in this tutorial).
path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))

# First, lets load a data-set using the 'load_imaging_from_fits' function of the image module (import as 'im). This
# data-set represents a data-reduction that is performed - it conforms to all the standard I will describe in this
# tutorial!
image = im.load_imaging_from_fits(image_path=path+'data/image/image.fits',
                                  noise_map_path=path+'data/image/noise_map.fits',
                                  psf_path=path+'data/image/psf.fits', pixel_scale=0.1)
imaging_plotters.plot_image_subplot(image=image)

# If your data comes in one .fits file spread across multiple hdus, you can specify the hdus of each image instead.
image = im.load_imaging_from_fits(image_path=path+'data/image/multiple_hdus.fits', image_hdu=0,
                                  noise_map_path=path+'data/image/multiple_hdus.fits', noise_map_hdu=1,
                                  psf_path=path+'data/image/multiple_hdus.fits', psf_hdu=2, pixel_scale=0.1)
imaging_plotters.plot_image_subplot(image=image)

# Now, lets think about the format and data-reduction of our data. There are numerous reasons why the image we just
# looked at is a good data-set for lens modeling. I strongly recommend you reduce your data to conform to the
# standards discussed below - it'll make your time using PyAutoLens a lot simpler.

# However, you may not have access to the data-reduction tools that masde the data, and we've included a number of
# in-built functions in PyAutoLens to convert the data to a good format for you - but it's much easier if you can just
# reduce it this way in the first place!

# 1) Brightness units - the image aboves flux and noise-map values are in units of electrons per second (and not
#    electrons, counts, ADU's etc.). Although PyAutoLens can perform an analysis using other units, a number of default
#    settings assume the image is in electrons per second, for example the priors on light profile intensities and pixelization regularization.

# Lets look at an image that is in units of counts - its easy to tell because the peak values are in the 1000's or
# 10000's.
image = im.load_imaging_from_fits(image_path=path+'data/image_in_counts/image.fits', pixel_scale=0.1,
                                  noise_map_path=path+'data/image_in_counts/noise_map.fits',
                                  psf_path=path+'data/image_in_counts/psf.fits')
imaging_plotters.plot_image_subplot(image=image)

# If your image is in counts, you can convert it to electrons per second by supplying the function above with an
# exposure time and using the 'convert_arrays_from_counts' boolean flag.
image = im.load_imaging_from_fits(image_path=path+'data/image_in_counts/image.fits', pixel_scale=0.1,
                                  noise_map_path=path+'data/image_in_counts/noise_map.fits',
                                  psf_path=path+'data/image_in_counts/psf.fits',
                                  exposure_time=1000.0, convert_arrays_from_counts=True)
imaging_plotters.plot_image_subplot(image=image)

# The effective exposure time in each pixel may vary. This occurs when data is reduced in a specific way, often to
# remove effects like cosmic rays. If you have access to an effective exposure-time map, you can use this to convert
# the image to electrons per second instead.
image = im.load_imaging_from_fits(image_path=path+'data/image_in_counts/image.fits', pixel_scale=0.1,
                                  noise_map_path=path+'data/image_in_counts/noise_map.fits',
                                  psf_path=path+'data/image_in_counts/psf.fits',
                                  exposure_time_map_path=path+'data/image_in_counts/exposure_time_map.fits',
                                  convert_arrays_from_counts=True)
imaging_plotters.plot_image_subplot(image=image)


# 2) The lens galaxy is in the centre of the images, as opposed to a corner. This is convinient as it means the centre
#    of the lens galaxy's light and mass profiles will be near the origin (0.0", 0.0"), alongside the centre of the
#    image mask. By default, these are the coordinates the priors on these parameters assume.

# 3) The postage-stamp cut-out of the lens galaxy is sufficiently big that the mask will include all regions of interest
#    around the source and lens galaxy, but not so big that it requires a large amount of memory to store it.

# 4) The noise-map values are the RMS standard deviation in every pixel (and not the variances, HST WHT-map values,
#    etc.). You MUST be 100% certain that the noise map is the RMS standard deviations, or else your analysis will
#    be incorrect.

# 5) The PSF dimensions are odd x odd (21 x 21). It is important that the PSF dimensions are odd, because even-sized
#    PSF kernels introduce a half-pixel offset in the convolution routine, which can lead to systematics in the lens
#    model analysis.

# 6) The PSF is zoomed in around its central core, which is the most important region for strong lens modeling. By
#    default, the size of the PSF image is used to perform convolution. The larger this stamp, the longer this
#    convolution will take to run. In geneal, we would recommend the PSF size is 21 x 21.