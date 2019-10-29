from autolens.model.profiles import mass_profiles
from autolens.model.profiles.plotters import profile_plotters

grid = al.grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=4)

sis_mass_profile = mass_profiles.EllipticalIsothermal(
    centre=(1.0, 1.0), einstein_radius=1.6, axis_ratio=0.7
)

profile_plotters.plot_convergence(
    mass_profile=sis_mass_profile,
    grid=grid,
    plot_critical_curves=False,
    plot_caustics=False,
)

# galaxy = al.Galaxy(mass=sis_mass_profile, redshift=1)
#
# galaxy_plotters.plot_convergence(
#     galaxy=galaxy,
#     grid=grid,
#     plot_critical_curves=False,
#     plot_caustics=True,
# )
