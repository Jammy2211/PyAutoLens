from autolens.data.array import scaled_array
from autolens.data.array.plotters import array_plotters

import os

# In this example, we will demonstrate how the appearance of figures in PyAutoLens can be customized. To do this, we
# will use the the image of the strong lens slacs1430+4105 from a .fits file and plot it using the
# function autolens.data.plotters.array_plotters.plot_array.

# The customization functions demonstrated in this example are generic to any 2D array of data, and can therefore be
# applied to the plotting of noise-maps, PSF's, residual maps, chi-squared maps, etc.

# We have included the .fits data required for this example in the directory
# 'workspace/output/data/example/slacs1430+4105/'.

# First, lets setup the path to the .fits file of the image.
lens_name = 'slacs1430+4105'

path = '{}/../../../..'.format(os.path.dirname(os.path.realpath(__file__)))
file_path = path+'/data/example/'+lens_name+'/image.fits'.format(os.path.dirname(os.path.realpath(__file__)))

# Now, lets load this array as a scaled array. A scaled array is an ordinary NumPy array, but it also includes a pixel
# scale which allows us to convert the axes of the array to arc-second coordinates.
image = scaled_array.ScaledSquarePixelArray.from_fits_with_pixel_scale(file_path=file_path, hdu=0, pixel_scale=0.03)

# We can now use an array plotter to plot the array. We customize the plot as follows:

# 1) We make the array's figure size bigger than the default size (7,7).

# 2) Because the figure is bigger, we increase the size of the title, x and y labels / ticks from their default size of
#    16 to 24.

# 3) For the same reason, we increase the size of the colorbar ticks from the default value 10 to 20.
array_plotters.plot_array(array=image, figsize=(12,12), title='SLACS1430+4105 Image',
                          titlesize=24, xlabelsize=24, ylabelsize=24, xyticksize=24,
                          cb_ticksize=20)

# The colormap of the array can be changed to any of the standard matplotlib colormaps.
array_plotters.plot_array(array=image, title='SLACS1430+4105 Image', cmap='spring')

# We can change the x / y axis units from arc-seconds to kiloparsec, by inputting a kiloparsec to arcsecond conversion
# factor (for SLACS1430+4105, the lens galaxy is at redshift 0.285, corresponding to the conversion factor below).
array_plotters.plot_array(array=image, title='SLACS1430+4105 Image', units='kpc', kpc_per_arcsec=4.335)

# The matplotlib figure can be output to the hard-disk as a png, as follows.
array_plotters.plot_array(array=image, title='SLACS1430+4105 Image',
                          output_path=path+'/plotting/plots/', output_filename='array', output_format='png')

# The array itself can be output to the hard-disk as a .fits file.
array_plotters.plot_array(array=image,
                          output_path=path+'/plotting/plots/', output_filename='array', output_format='fits')