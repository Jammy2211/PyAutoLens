import autolens as al
from autolens.plotters import phase_plotters

mask = al.mask.circular(
    shape_2d=(200, 200),
    pixel_scales=0.05,
    sub_size=1,
    radius_arcsec=2.4,
    centre=(2.0, 2.0),
)

grid = al.grid.from_mask(mask=mask)

lens_galaxy = al.galaxy(
    redshift=0.5,
    light=al.lp.EllipticalDevVaucouleurs(intensity=1.0),
    mass=al.mp.SphericalIsothermal(centre=(2.0, 2.0), einstein_radius=1.0),
)

source_galaxy = al.galaxy(
    redshift=1.0, light=al.lp.EllipticalExponential(centre=(2.0, 2.0), intensity=1.0)
)

tracer = al.tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# al.plot.plane.profile_image(plane=tracer.source_plane, grid=grid, include_critical_curves=True)

al.plot.tracer.subplot(
    tracer=tracer,
    grid=grid,
    mask=mask,
    include_critical_curves=False,
    include_caustics=True,
    positions=[[[3.0, 2.0], [2.0, 3.0]]],
)

# phase_plotters.ray_tracing_of_phase(tracer=tracer, grid=grid,     during_analysis=True,
#     mask=None,
#     include_critical_curves,
#     include_caustics,
#     positions,
#     units,
#     plot_as_subplot,
#     plot_all_at_end_png,
#     plot_all_at_end_fits,
#     plot_image,
#     plot_source_plane,
#     plot_convergence,
#     plot_potential,
#     plot_deflections,
#     visualize_path,
#     subplot_path,)

# galaxy = al.galaxy(mass=sis_mass_profile, redshift=1)
#
# al.plot.galaxy.convergence(
#     galaxy=galaxy,
#     grid=grid,
#     include_critical_curves=False,
#     include_caustics=True,
# )
