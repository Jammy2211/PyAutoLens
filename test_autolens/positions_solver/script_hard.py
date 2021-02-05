import autolens as al
import autolens.plot as aplt
import numpy as np

# mask = al.Mask2D.circular_annular(shape_native=(200, 200), pixel_scales=0.05, inner_radius=0.3, outer_radius=3.0)
# grid = al.Grid2D.from_mask(mask=mask)

grid = al.Grid2D.uniform(shape_native=(2500, 2500), pixel_scales=0.0002)

mass_profile = al.mp.EllipticalPowerLaw(
    centre=(0.001, 0.001), elliptical_comps=(0.5, 0.5), einstein_radius=1.0, slope=1.6
)

light_profile = al.lp.EllipticalExponential(
    centre=(0.1, 0.0), intensity=0.1, effective_radius=0.1
)

solver = al.PositionsSolver(
    grid=grid,
    use_upscaling=False,
    pixel_scale_precision=0.001,
    magnification_threshold=0.0,
)

positions = solver.solve(
    lensing_obj=mass_profile, source_plane_coordinate=light_profile.centre
)

lens_galaxy = al.Galaxy(redshift=0.5, mass=mass_profile)
source_galaxy = al.Galaxy(redshift=1.0, light=light_profile)
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

visuals_2d = aplt.Visuals2D(positions=positions)

magnification = tracer.magnification_from_grid(grid=grid)
magnification = np.nan_to_num(magnification)
print(magnification)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, visuals_2d=visuals_2d)
tracer_plotter.figures(image=True)


mat_plot_2d = aplt.MatPlot2D(
    cmap=aplt.Cmap(vmax=9.0, vmin=-9.0),
    positions_scatter=aplt.PositionsScatter(s=100, c="w"),
)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, visuals_2d=visuals_2d, mat_plot_2d=mat_plot_2d
)
tracer_plotter.figure_magnification()
