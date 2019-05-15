from autolens.data.array import grids
from autolens.model.profiles import mass_profiles as mp

# This numerics test determines the range of plausible values a power-law mass profile can take and not crash due to
# numerical integration. This informs us the limits config file for this profile.

# In this test, we'll pass a regular grid to a power-law profile and compute deflection angles. We will check if
# the pyquad integrator crashes for certain values of (y,x) coordinates.
regular_grid = grids.RegularGrid.from_shape_and_pixel_scale(shape=(25, 25), pixel_scale=0.005)

y_centres = [0.01, 0.005, 0.001, 0.0001, 0.00000001, 0.0]

for y in y_centres:

    print('Normal Power-Law (centre offset = {:.8f})'.format(y))
    power_law = mp.EllipticalPowerLaw(centre=(y, 0.0), axis_ratio=0.8, phi=0.0, einstein_radius=1.0, slope=2.0)
    power_law.deflections_from_grid(grid=regular_grid, grid_radial_minimum=1.0e-8)

for y in y_centres:

    print('Axis Ratio 1.0 Power-Law (centre offset = {:.8f})'.format(y))
    power_law = mp.EllipticalPowerLaw(centre=(y, 0.0), axis_ratio=1.0, phi=0.0, einstein_radius=1.0, slope=2.0)
    power_law.deflections_from_grid(grid=regular_grid, grid_radial_minimum=1.0e-8)

for y in y_centres:

    print('Axis Ratio 0.01 Power-Law (centre offset = {:.8f})'.format(y))
    power_law = mp.EllipticalPowerLaw(centre=(y, 0.0), axis_ratio=0.005, phi=0.0, einstein_radius=1.0, slope=2.0)
    power_law.deflections_from_grid(grid=regular_grid, grid_radial_minimum=1.0e-8)

for y in y_centres:

    print('Slope Low Power-Law (centre offset = {:.8f})'.format(y))
    power_law = mp.EllipticalPowerLaw(centre=(y, 0.0), axis_ratio=0.8, phi=0.0, einstein_radius=1.0, slope=-10.0)
    power_law.deflections_from_grid(grid=regular_grid, grid_radial_minimum=1.0e-8)

for y in y_centres:

    print('Slope High Normal Power-Law (centre offset = {:.8f})'.format(y))
    power_law = mp.EllipticalPowerLaw(centre=(y, 0.0), axis_ratio=0.8, phi=0.0, einstein_radius=1.0, slope=3.0)
    power_law.deflections_from_grid(grid=regular_grid, grid_radial_minimum=1.0e-8)