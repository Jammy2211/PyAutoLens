import autolens as al
from autolens.plots import phase_plots

mask = al.mask.circular(
    shape_2d=(200, 200), pixel_scales=0.03, sub_size=1, radius=2.4, centre=(0.0, 0.0)
)

grid = al.grid.from_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.01, 0.0), einstein_radius=1.0, axis_ratio=0.8
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalExponential(
        centre=(0.02, 0.01), intensity=1.0, effective_radius=0.01
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

print(tracer.image_plane_multiple_image_positions_of_galaxies(grid=grid))

# al.plot.plane.profile_image(plane=tracer.source_plane, grid=grid, include_critical_curves=True)

al.plot.tracer.profile_image(
    tracer=tracer,
    grid=grid,
    mask=mask,
    include_multiple_images=True,
    include_critical_curves=False,
    include_caustics=True,
)

# phase_plots.ray_tracing_of_phase(tracer=tracer, grid=grid,     during_analysis=True,
#     mask=None,
#     include_critical_curves,
#     include_caustics,
#     positions,
#     unit_label,
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

# galaxy = al.Galaxy(mass=sis_mass_profile, redshift=1)
#
# al.plot.galaxy.convergence(
#     galaxy=galaxy,
#     grid=grid,
#     include_critical_curves=False,
#     include_caustics=True,
# )
