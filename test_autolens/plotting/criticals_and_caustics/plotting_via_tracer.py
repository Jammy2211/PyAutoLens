import autolens as al
from autolens.plotters import phase_plotters

mask = al.mask.circular(shape_2d=(100, 100), pixel_scales=0.05, sub_size=1, radius_arcsec=2.4)

grid = al.grid.from_mask(mask=mask)

lens_galaxy = al.galaxy(
    redshift=0.5,
    light=al.lp.EllipticalDevVaucouleurs(intensity=1.0),
    mass=al.mp.SphericalIsothermal(
    centre=(0.0, 0.0), einstein_radius=1.6)
)

source_galaxy = al.galaxy(
    redshift=1.0,
    light=al.lp.EllipticalExponential(intensity=1.0))

tracer = al.tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# al.plot.plane.profile_image(plane=tracer.source_plane, grid=grid, plot_critical_curves=True)

al.plot.tracer.subplot(tracer=tracer, grid=grid, plot_critical_curves=True, plot_caustics=True)

# phase_plotters.ray_tracing_of_phase(tracer=tracer, grid=grid,     during_analysis=True,
#     mask=None,
#     plot_critical_curves,
#     plot_caustics,
#     positions,
#     units,
#     should_plot_as_subplot,
#     should_plot_all_at_end_png,
#     should_plot_all_at_end_fits,
#     should_plot_image,
#     should_plot_source_plane,
#     should_plot_convergence,
#     should_plot_potential,
#     should_plot_deflections,
#     visualize_path,
#     subplot_path,)

# galaxy = al.galaxy(mass=sis_mass_profile, redshift=1)
#
# al.plot.galaxy.convergence(
#     galaxy=galaxy,
#     grid=grid,
#     plot_critical_curves=False,
#     plot_caustics=True,
# )
