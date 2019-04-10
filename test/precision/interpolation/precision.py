from autolens.data.array import grids
from autolens.model.profiles import mass_profiles as mp
from autolens.lens import lens_data as ld
from autolens.data.array import mask as msk

from autolens.plotters import array_plotters

from test.simulation import simulation_util

import numpy as np

# Although we could test the deflection angles without using an image (e.g. by just making a grid), we have chosen to
# set this test up using an image and mask. This gives run-time numbers that can be easily related to an actual lens
# analysis

sub_grid_size = 2
inner_radius_arcsec = 0.0
outer_radius_arcsec = 4.0

print('sub grid size = ' + str(sub_grid_size))
print('annular inner mask radius = ' + str(inner_radius_arcsec) + '\n')
print('annular outer mask radius = ' + str(outer_radius_arcsec) + '\n')

for data_resolution in ['HST_Up']:

    print()

    ccd_data = simulation_util.load_test_ccd_data(data_type='no_lens_source_smooth', data_resolution=data_resolution,
                                                  psf_shape=(3, 3))
    mask = msk.Mask.circular_annular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale,
                                     inner_radius_arcsec=inner_radius_arcsec, outer_radius_arcsec=outer_radius_arcsec)
    lens_data = ld.LensData(ccd_data=ccd_data, mask=mask, sub_grid_size=sub_grid_size)

    print('Deflection angle run times for image type ' + data_resolution + '\n')
    print('Number of points = ' + str(lens_data.grid_stack.regular.shape[0]) + '\n')

    interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(mask=lens_data.mask,
                                                                             grid=lens_data.grid_stack.sub,
                                                                             interp_pixel_scale=0.1)

    print('Number of interpolation points = ' + str(interpolator.interp_grid.shape[0]) + '\n')

    ### EllipticalIsothermal ###

    mass_profile = mp.EllipticalCoredIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=0.5,
                                                core_radius=0.3)

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

    difference_y_2d = lens_data.grid_stack.sub.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
        sub_array_1d=difference_y)
    difference_x_2d = lens_data.grid_stack.sub.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
        sub_array_1d=difference_x)

    array_plotters.plot_array(array=difference_y_2d)