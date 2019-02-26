from autolens.data.array import scaled_array
from autolens.data.array.plotters import array_plotters

import os

# In this example, we will load the image of a strong lens from a .fits file and plot it using the
# function autolens.data.plotters.array_plotters.plot_array. We will customize the appearance of this figure to
# highlight the features of the image. For more generical plotting tools (e.g. changing the figure size, axis units,
# outputting the image to the hard-disk, etc.) checkout the example in 'workspace/plotting/examples/arrays/array.py'.

# We will use the image of slacs1430+4105.

# We have included the .fits data required for this example in the directory
# 'workspace/output/data/example/slacs1430+4105/'.

# First, lets setup the path to the .fits file of the image.
lens_name = 'slacs1430+4105'

path = '{}/../../..'.format(os.path.dirname(os.path.realpath(__file__)))
image_path = path + '/data/example/' + lens_name + '/image.fits'

# Now, lets load this image as a scaled array. A scaled array is an ordinary NumPy array, but it also includes a pixel
# scale which allows us to convert the axes of the image to arc-second coordinates.
image = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=image_path, hdu=0, pixel_scale=0.03)

# We can now use an array plotter to plot the image. Lets first plot it using the default PyAutoLens settings.
array_plotters.plot_array(array=image, title='SLACS1430+4105 Image')

# For a lens like SLACS1430+4105, the lens galaxy's light outshines the background source, making it appear faint.
# we can use a symmetric logarithmic colorbar normalization to better reveal the source galaxy (due to negative values
# in the image, we cannot use a regular logirithmic colorbar normalization).
array_plotters.plot_array(array=image, title='SLACS1430+4105 Image',
                          norm='symmetric_log', linthresh=0.05, linscale=0.02)

# Alternatively, we can use the default linear colorbar normalization and customize the limits over which the colormap
# spans its dynamic range.
array_plotters.plot_array(array=image, title='SLACS1430+4105 Image',
                          norm='linear', norm_min=0.0, norm_max=0.3)

# We can also load the full set of ccd data (image, noise-map, PSF) and use the ccd_plotters to make the figures above.
from autolens.data import ccd
from autolens.data.plotters import ccd_plotters

psf_path = path + '/data/example/' + lens_name + '/psf.fits'
noise_map_path = path + '/data/example/' + lens_name + '/noise_map.fits'

ccd_data = ccd.load_ccd_data_from_fits(image_path=image_path, psf_path=psf_path, noise_map_path=noise_map_path,
                                       pixel_scale=0.03)

# These plotters can be customized using the exact same functions as above.
ccd_plotters.plot_noise_map(ccd_data=ccd_data, title='SLACS1430+4105 Noise-Map',
                            norm='log')

# Of course, as we've seen in many other examples, a sub-plot of the ccd data can be plotted. This can also take the
# customization inputs above, but it should be noted that the options are applied to all images, and thus will most
# likely degrade a number of the sub-plot images.
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, norm='symmetric_log', linthresh=0.05, linscale=0.02)