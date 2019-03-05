import time

from autolens.data.array import grids
from autolens.model.profiles import mass_profiles as mp
from autolens.lens import lens_data as ld
from autolens.data.array import mask as msk

from autolens.data.array.plotters import array_plotters

from test.profiling import tools

import numpy as np
import matplotlib.pyplot as plt

# Although we could test the deflection angles without using an image (e.g. by just making a grid), we have chosen to
# set this test up using an image and mask. This gives run-time numbers that can be easily related to an actual lens
# analysis

sub_grid_size = 2
inner_radius_arcsec = 0.0
outer_radius_arcsec = 4.0

print('sub grid size = ' + str(sub_grid_size))
print('annular inner mask radius = ' + str(inner_radius_arcsec) + '\n')
print('annular outer mask radius = ' + str(outer_radius_arcsec) + '\n')

for image_type in ['HST_Up']:

    print()

    ccd_data = tools.load_profiling_ccd_data(image_type=image_type, lens_name='no_lens_source_smooth', psf_shape=(3,3))
    mask = msk.Mask.circular_annular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale,
                                     inner_radius_arcsec=inner_radius_arcsec, outer_radius_arcsec=outer_radius_arcsec)
    lens_data = ld.LensData(ccd_data=ccd_data, mask=mask, sub_grid_size=sub_grid_size)

    print('Deflection angle run times for image type ' + image_type + '\n')
    print('Number of points = ' + str(lens_data.grid_stack.regular.shape[0]) + '\n')

    interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(mask=lens_data.mask,
                                                                             grid=lens_data.grid_stack.sub,
                                                                             interp_pixel_scale=1.0)

    print('Number of interpolation points = ' + str(interpolator.interp_grid.shape[0]) + '\n')

    ### EllipticalIsothermal ###

    mass_profile = mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=45.0, einstein_radius=0.5)

    interp_deflections = mass_profile.deflections_from_grid(grid=interpolator.interp_grid)
    deflections = np.zeros((lens_data.grid_stack.sub.shape[0], 2))
    deflections[:,0] = interpolator.interpolated_values_from_values(values=interp_deflections[:,0])
    deflections[:,1] = interpolator.interpolated_values_from_values(values=interp_deflections[:,1])

    true_deflections = mass_profile.deflections_from_grid(grid=lens_data.grid_stack.sub)

    difference_y = deflections[:,0] - true_deflections[:, 0]
    difference_x = deflections[:,1] - true_deflections[:, 1]

    print("interpolation y error: ", np.mean(difference_y))
    print("interpolation y uncertainty: ", np.std(difference_y))
    print("interpolation y max error: ", np.max(difference_y))
    print("interpolation x error: ", np.mean(difference_x))
    print("interpolation x uncertainty: ", np.std(difference_x))
    print("interpolation x max error: ", np.max(difference_x))

    difference_y_2d = lens_data.grid_stack.regular.scaled_array_from_array_1d(array_1d=difference_y)
    difference_x_2d = lens_data.grid_stack.regular.scaled_array_from_array_1d(array_1d=difference_x)

    array_plotters.plot_array(array=difference_y_2d)#, grid=interpolator.interp_grid)