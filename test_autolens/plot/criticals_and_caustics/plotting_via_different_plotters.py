import autolens as al

grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=4)

sis_mass_profile = al.mp.EllipticalIsothermal(
    centre=(1.0, 1.0), einstein_radius=1.6, axis_ratio=0.7
)

aplt.mp.convergence(
    mass_profile=sis_mass_profile,
    grid=grid,
    include_critical_curves=False,
    include_caustics=False,
)

# galaxy = al.Galaxy(mass=sis_mass_profile, redshift=1)
#
# aplt.galaxy.convergence(
#     galaxy=galaxy,
#     grid=grid,
#     include_critical_curves=False,
#     include_caustics=True,
# )
