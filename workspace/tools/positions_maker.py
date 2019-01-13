from autolens.data import ccd
from autolens.data.plotters import data_plotters

import os

# This tool allows one to input a set of positions of a multiply imaged strongly lensed source, corresponding to a set
# positions / pixels which are anticipated to trace to the same location in the source-plane.

# A non-linear sampler uses these positions to discard the mass-models where they do not trace within a threshold of
# one another, speeding up the analysis and removing unwanted solutions with too much / too little mass.

# The 'lens name' is the name of the lens in the data folder, e.g. the positions will be output as 
# '/workspace/data/example_sim/positionss.dat' (this file is already in the workspace and is remade running this script)
lens_name = 'example_lens_light_and_x1_source'
pixel_scale = 0.1 # If you use this tool for your own data, you *must* double check this pixel scale is correct!

# First, load the CCD imaging data, so that the positions can be plotted over the strong lens image.
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))
image = ccd.load_image(image_path=path+'/data/'+lens_name+'/image.fits', image_hdu=0, pixel_scale=pixel_scale)

# Now, create a set of positions, which is simply a python list of (y,x) values.
positions = [[[0.8, 1.45], [1.78, -0.4], [-0.95, 1.38], [-0.83, -1.04]]]

# We can infact input multiple lists of positions (commented out below), which corresponds to pixels which are \
# anticipated to map to different multiply imaged regions of the source-plane (e.g. you would need something like \
# spectra to be able to do this)
                  # Images of source 1           # Images of source 2
# positions = [[[1.0, 1.0], [2.0, 0.5]], [[-1.0, -0.1], [2.0, 2.0], [3.0, 3.0]]]]

# Now lets plot the image and positions, so we can check that the positions overlap different regions of the source.
data_plotters.plot_image(image=image, positions=positions)

# Now we're happy with the positions, lets output them to the data folder of the lens, so that we can load them from a
# .dat file in our pipelines!
ccd.output_positions(positions=positions, positions_path=path+'/data/'+lens_name+'/positions.dat')


# These commented out lines would create the positions for the example_multi_plane data.
# lens_name = 'example_multi_plane'
# pixel_scale = 0.05
# positions = [[[0.8, 1.12], [-0.64, 1.13], [1.38, -0.2], [-0.72, -0.83]]]