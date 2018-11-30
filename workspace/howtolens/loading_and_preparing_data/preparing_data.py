import os

from autolens.data.imaging import image as im
from autolens.data.array import mask as ma
from autolens.lensing import lensing_image as li
from autolens.data.imaging.plotters import imaging_plotters
from workspace.howtolens.loading_and_preparing_data import simulate_data

# To model data with PyAutoLens, you first need to ensure it is in a format suitable for lens modeling. This tutorial
# takes you through what format to use, and will introduce a number of PyAutoLens's built in tools that can convert
# data from an unsuitable format.

# First, lets setup the path to our current working directory. I recommend you use the 'AutoLens/workspace' directory
# and that you place your data in the 'AutoLens/workspace/data' directory.

# (for this tutorial, we'll use the 'AutoLens/workspace/howtolens/preparing_data' directory. The folder 'data' contains
# the example data-sets we'll use in this tutorial).
path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))
simulate_data.simulate_all_images() # This will populate the 'data' folder.

# First, lets load a data-set using the 'load_imaging_from_fits' function of the regular module (import as 'im'). This
# data-set represents a good data-reduction - it conforms to all the formatting standards I describe in this tutorial!
image = im.load_imaging_from_fits(image_path=path+'data/regular/regular.fits',
                                  noise_map_path=path+'data/regular/noise_map.fits',
                                  psf_path=path+'data/regular/psf.fits', pixel_scale=0.1)
imaging_plotters.plot_image_subplot(image=image)

# If your data comes in one .fits file spread across multiple hdus, you can specify the hdus of each regular instead.
image = im.load_imaging_from_fits(image_path=path+'data/regular/multiple_hdus.fits', image_hdu=0,
                                  noise_map_path=path+'data/regular/multiple_hdus.fits', noise_map_hdu=1,
                                  psf_path=path+'data/regular/multiple_hdus.fits', psf_hdu=2, pixel_scale=0.1)
imaging_plotters.plot_image_subplot(image=image)

# Now, lets think about the format and data-reduction of our data. There are numerous reasons why the regular we just
# looked at is a good data-set for lens modeling. I strongly recommend you reduce your data to conform to the
# standards discussed below - it'll make your time using PyAutoLens a lot simpler.

# However, you may not have access to the data-reduction tools that made the data, so we've included a number of
# in-built functions in PyAutoLens to convert the data to a good format for you. However, your life will be much easier
# if you can just reduce it this way in the first place!

# 1) Brightness units - the regular's flux and noise-map values are in units of electrons per second (and not electrons,
#    counts, ADU's etc.). Although PyAutoLens can technically perform an analysis using other units, the default
#    settings assume the regular is in electrons per second (e.g. the priors on light profile intensities and
#    regularization coefficient). Thus, images not in electrons per second should be converted!

# Lets look at an regular that is in units of counts - its easy to tell because the peak values are in the 1000's or
# 10000's.
image_in_counts = im.load_imaging_from_fits(image_path=path+'data/image_in_counts/regular.fits', pixel_scale=0.1,
                                  noise_map_path=path+'data/image_in_counts/noise_map.fits',
                                  psf_path=path+'data/image_in_counts/psf.fits')
imaging_plotters.plot_image_subplot(image=image_in_counts)

# If your image is in counts, you can convert it to electrons per second by supplying the function above with an
# exposure time and using the 'convert_arrays_from_counts' boolean flag.
image_converted_to_eps = im.load_imaging_from_fits(image_path=path+'data/image_in_counts/regular.fits', pixel_scale=0.1,
                                                   noise_map_path=path+'data/image_in_counts/noise_map.fits',
                                                   psf_path=path+'data/image_in_counts/psf.fits',
                                                   exposure_time_map_from_single_value=1000.0, convert_from_counts=True)
imaging_plotters.plot_image_subplot(image=image_converted_to_eps)

# The effective exposure time in each pixel may vary. This occurs when data is reduced in a specific way, called
# 'dithering' and 'drizzling'. If you have access to an effective exposure-time map, you can use this to convert
# the regular to electrons per second instead.
image_converted_to_eps = im.load_imaging_from_fits(image_path=path+'data/image_in_counts/regular.fits', pixel_scale=0.1,
                                                   noise_map_path=path+'data/image_in_counts/noise_map.fits',
                                                   psf_path=path+'data/image_in_counts/psf.fits',
                                                   exposure_time_map_path=path+'data/image_in_counts/exposure_time_map.fits',
                                                   convert_from_counts=True)
imaging_plotters.plot_image_subplot(image=image_converted_to_eps)

# 2) Postage stamp size - The bigger the postage stamp cut-out of the regular, the more memory it requires to store it.
#    Why keep the edges surrounding the lens if there is no actual signal there?

#    Lets look at an example of a very large postage stamp - we can barely even see the lens and source galaxies!
image_large_stamp = im.load_imaging_from_fits(image_path=path+'data/image_large_stamp/regular.fits', pixel_scale=0.1,
                                              noise_map_path=path+'data/image_large_stamp/noise_map.fits',
                                              psf_path=path+'data/image_large_stamp/psf.fits')
imaging_plotters.plot_image_subplot(image=image_large_stamp)

#    If you have a large postage stamp, you can trim it when you load the data by specifying a new regular size in pixels.
#    This will also trim the noise-map, exposoure time map and other arrays which are the same dimensions / scale as
#    the regular. This trimming is centred on the regular.
image_large_stamp_trimmed = im.load_imaging_from_fits(image_path=path+'data/image_large_stamp/regular.fits',
                                                      pixel_scale=0.1,
                                                      noise_map_path=path+'data/image_large_stamp/noise_map.fits',
                                                      psf_path=path+'data/image_large_stamp/psf.fits',
                                                      resized_image_shape=(101, 101))
imaging_plotters.plot_image_subplot(image=image_large_stamp_trimmed)

# 3) Postage stamp size - On the other hand, the postage stamp must have enough padding in the border that our mask can
#    include all pixels with signal in. In fact, it isn't just the mask that must be contained within the postage stamp,
#    but also the mask's 'blurring region' - which corresponds to all unmasked regular pixels where light will blur into
#    the mask after PSF convolution. Thus, we may need to pad an regular to include this region.

#    This regular is an example of a stamp which is big enough to contain the lens and source galaxies, but when we
#    apply a sensible mask we get an error, because the mask's blurring region hits the edge of the regular.
image_small_stamp = im.load_imaging_from_fits(image_path=path+'data/image_small_stamp/regular.fits', pixel_scale=0.1,
                                              noise_map_path=path+'data/image_small_stamp/noise_map.fits',
                                              psf_path=path+'data/image_small_stamp/psf.fits')
imaging_plotters.plot_image_subplot(image=image_small_stamp)

# If we apply a mask to this regular, we'll get an error when we try to use it to set up a lensing regular, because its
# blurring region hits the regular edge.
mask = ma.Mask.circular(shape=image_small_stamp.shape, pixel_scale=image_small_stamp.pixel_scale,
                        radius_mask_arcsec=2.0)
# lensing_image = li.LensingImage(regular=regular, mask=mask)

# We can overcome this using the same input as before. However, now, the resized regular shape is bigger than the regular,
# thus a padding of zeros is introduced to the edges.
image_small_stamp_padded = im.load_imaging_from_fits(image_path=path+'data/image_small_stamp/regular.fits',
                                                     pixel_scale=0.1,
                                                     noise_map_path=path+'data/image_small_stamp/noise_map.fits',
                                                     psf_path=path+'data/image_small_stamp/psf.fits',
                                                     resized_image_shape=(140, 140))
mask = ma.Mask.circular(shape=image_small_stamp_padded.shape, pixel_scale=image_small_stamp_padded.pixel_scale,
                        radius_mask_arcsec=2.0)
imaging_plotters.plot_image_subplot(image=image_small_stamp_padded, mask=mask)
lensing_image = li.LensingImage(image=image_small_stamp_padded, mask=mask)

########## IVE INCLUDED THE TEXT FOR 5 BELOW SO YOU CAN BE AWARE OF CENTERING, BUT THE BUILT IN FUNCTIONALITY FOR #####
########## RECENTERING CURRENTLY DOES NOT WORK :( ###########

# 5) Lens Galaxy Centering - The regular should place the lens galaxy in the origin of the regular, as opposed to a
#    corner. This ensures the origin of the lens galaxy's light and mass profiles will be near the origin (0.0", 0.0"),
#    as wel as the origin of the mask, which is a more intuitive coordinate system. The priors on the light
#    profiles and mass profile also assume a origin of (0.0", 0.0"), as well as the default mask origin.

# Lets look at an off-center regular - clearly both the lens galaxy and Einstein ring are offset in the positive y and x d
# directions.
# image_offset_centre = im.load_imaging_from_fits(image_path=path+'data/image_offset_centre/regular.fits', pixel_scale=0.1,
#                                   noise_map_path=path+'data/image_offset_centre/noise_map.fits',
#                                   psf_path=path+'data/image_offset_centre/psf.fits')
# imaging_plotters.plot_image_subplot(regular=image_offset_centre)

# We can address this by using supplying a new origin for the regular, in pixels. We also supply the resized shape, to
# instruct the code whether it should trim the regular or pad the edges that now arise due to recentering.
# image_recentred_pixels = im.load_imaging_from_fits(image_path=path+'data/image_small_stamp/regular.fits', pixel_scale=0.1,
#                                             noise_map_path=path+'data/image_small_stamp/noise_map.fits',
#                                             psf_path=path+'data/image_small_stamp/psf.fits',
#                                             resized_image_shape=(100, 100),
#                                             resized_image_centre_pixels=(0, 0))
# #                                            resized_image_centre_arc_seconds=(1.0, 1.0))
# print(image_recentred_pixels.shape)
# imaging_plotters.plot_image_subplot(regular=image_recentred_pixels)

# 6) The noise-map values are the RMS standard deviation in every pixel (and not the variances, HST WHT-map values,
#    etc.). You MUST be 100% certain that the noise map is the RMS standard deviations, or else your analysis will
#    be incorrect.

# There are many different ways the noise-map can be reduced. We are aiming to include conversion functions for all
# common data-reductions. Currently, we have a function to convert an regular from a HST WHT map, where
# RMS SD = 1.0/ sqrt(WHT). This can be called using the 'convert_noise_map_from_weight_map' flag.
image_noise_from_wht = im.load_imaging_from_fits(image_path=path+'data/image_large_stamp/regular.fits',
                                                 pixel_scale=0.1,
                                                 noise_map_path=path+'data/image_large_stamp/noise_map.fits',
                                                 psf_path=path+'data/image_large_stamp/psf.fits',
                                                 convert_noise_map_from_weight_map=True)

# (I don't currently have an example regular in WHT for this tutorial, but the function above will work. Above, it
# actually converts an accurate noise-map to an inverse WHT map!

# 7) The PSF is zoomed in around its central core, which is the most important region for strong lens modeling. By
#    default, the size of the PSF regular is used to perform convolution. The larger this stamp, the longer this
#    convolution will take to run. In geneal, we would recommend the PSF size is 21 x 21.

#    Lets look at an regular where a large PSF kernel is loaded.
image_with_large_psf = im.load_imaging_from_fits(image_path=path+'data/image_with_large_psf/regular.fits',
                                                 pixel_scale=0.1,
                                                 noise_map_path=path+'data/image_with_large_psf/noise_map.fits',
                                                 psf_path=path+'data/image_with_large_psf/psf.fits')
imaging_plotters.plot_image_subplot(image=image_with_large_psf)

# We can resize a psf the same way that we resize an regular.
image_with_trimmed_psf = im.load_imaging_from_fits(image_path=path+'data/image_with_large_psf/regular.fits',
                                                 pixel_scale=0.1,
                                                 noise_map_path=path+'data/image_with_large_psf/noise_map.fits',
                                                 psf_path=path+'data/image_with_large_psf/psf.fits',
                                                   resized_psf_shape=(21, 21))
imaging_plotters.plot_image_subplot(image=image_with_trimmed_psf)

# 8) The PSF dimensions are odd x odd (21 x 21). It is important that the PSF dimensions are odd, because even-sized
#    PSF kernels introduce a half-pixel offset in the convolution routine, which can lead to systematics in the lens
#    model analysis.

# We do not currently yet have built-in functionality to address this issue. Therefore, if your PSF has an even
# dimension, you must manually trim and recentre it. If you need help on doing this, contact me on the PyAutoLens
# SLACK channel, as I'll have already written the routine to do this by the time you read this tutorial!