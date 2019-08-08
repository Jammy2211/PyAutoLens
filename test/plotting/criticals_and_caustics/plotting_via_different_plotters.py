from autolens.model.profiles import light_profiles
from autolens.model.profiles import mass_profiles
from autolens.data.array import grids
from autolens.model.profiles.plotters import profile_plotters

grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(
    shape=(100, 100), pixel_scale=0.05, sub_grid_size=8
)

sis_mass_profile = mass_profiles.EllipticalIsothermal(
    centre=(0.1, 0.1), einstein_radius=1.6, axis_ratio=0.7
)

profile_plotters.plot_convergence(
    mass_profile=sis_mass_profile, grid=grid_stack.regular, plot_critical_curves=True, plot_caustics=True
)
