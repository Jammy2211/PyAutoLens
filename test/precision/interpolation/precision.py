from autolens.array import grids, mask as msk
from autolens.model.profiles import mass_profiles as mp
from autolens.lens import lens_data as ld

from autolens.plotters import array_plotters

from test.simulation import simulation_util

import numpy as np

# Although we could test the deflection angles without using an image (e.al. by just making a grid), we have chosen to
# set this test up using an image and mask. This gives run-time numbers that can be easily related to an actual lens
# analysis

sub_size = 2
inner_radius_arcsec = 0.2
outer_radius_arcsec = 4.0

print("sub grid size = " + str(sub_size))
print("annular inner mask radius = " + str(inner_radius_arcsec) + "\n")
print("annular outer mask radius = " + str(outer_radius_arcsec) + "\n")

for data_resolution in ["HST_Up"]:

    print()

    imaging_data = simulation_util.load_test_imaging_data(
        data_type="lens_mass__source_smooth",
        data_resolution=data_resolution,
        psf_shape=(3, 3),
    )
    mask = al.Mask.circular_annular(
        shape=imaging_data.shape,
        pixel_scale=imaging_data.pixel_scale,
        inner_radius_arcsec=inner_radius_arcsec,
        outer_radius_arcsec=outer_radius_arcsec,
    )
    lens_data = al.LensData(imaging_data=imaging_data, mask=mask, sub_size=sub_size)

    print("Deflection angle run times for image type " + data_resolution + "\n")
    print("Number of points = " + str(lens_data.grid.shape[0]) + "\n")

    interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
        mask=lens_data.mask, grid=lens_data.grid, pixel_scale_interpolation_grid=0.05
    )

    print(
        "Number of interpolation points = "
        + str(interpolator.interp_grid.shape[0])
        + "\n"
    )

    ### EllipticalIsothermal ###

    mass_profile = al.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.0
    )

    interp_deflections = mass_profile.deflections_from_grid(
        grid=interpolator.interp_grid
    )
    deflections = np.zeros((lens_data.grid.shape[0], 2))
    deflections[:, 0] = interpolator.interpolated_values_from_values(
        values=interp_deflections[:, 0]
    )
    deflections[:, 1] = interpolator.interpolated_values_from_values(
        values=interp_deflections[:, 1]
    )

    true_deflections = mass_profile.deflections_from_grid(grid=lens_data.grid)

    true_deflections_y_2d = lens_data.grid.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
        sub_array_1d=true_deflections[:, 0]
    )
    true_deflections_x_2d = lens_data.grid.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
        sub_array_1d=true_deflections[:, 1]
    )

    difference_y = deflections[:, 0] - true_deflections[:, 0]
    difference_x = deflections[:, 1] - true_deflections[:, 1]

    print("interpolation y error: ", np.mean(difference_y))
    print("interpolation y uncertainty: ", np.std(difference_y))
    print("interpolation y max error: ", np.max(difference_y))
    print("interpolation x error: ", np.mean(difference_x))
    print("interpolation x uncertainty: ", np.std(difference_x))
    print("interpolation x max error: ", np.max(difference_x))

    difference_y_2d = lens_data.grid.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
        sub_array_1d=difference_y
    )
    difference_x_2d = lens_data.grid.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
        sub_array_1d=difference_x
    )

    array_plotters.plot_array(array=true_deflections_y_2d)
    array_plotters.plot_array(array=difference_y_2d)

    array_plotters.plot_array(array=true_deflections_x_2d)
    array_plotters.plot_array(array=difference_x_2d)

    # difference_percent_y = (np.abs(difference_y) / np.abs(true_deflections[:,0]))*100.0
    # difference_percent_x = (np.abs(difference_x) / np.abs(true_deflections[:,1]))*100.0
    #
    # print("interpolation y mean percent difference: ", np.mean(difference_percent_y))
    # print("interpolation y std percent difference: ", np.std(difference_percent_y))
    # print("interpolation y max percent difference: ", np.max(difference_percent_y))
    # print("interpolation x mean percent difference: ", np.mean(difference_percent_x))
    # print("interpolation x std percent difference: ", np.std(difference_percent_x))
    # print("interpolation x mean percent difference: ", np.max(difference_percent_x))
    #
    # difference_percent_y_2d = lens_data.grid.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
    #     sub_array_1d=difference_percent_y)
    # difference_percent_x_2d = lens_data.grid.scaled_array_2d_with_sub_dimensions_from_sub_array_1d(
    #     sub_array_1d=difference_percent_x)
    #
    # array_plotters.plot_array(array=difference_percent_y_2d)
    # array_plotters.plot_array(array=difference_percent_x_2d)
