import autolens as al
import autolens.plot as aplt

plotter = aplt.MatPlot2D()
plotter = aplt.MatPlot2D()

grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.0, elliptical_comps=(0.1, 0.1)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalExponential(
        centre=(0.02, 0.01), intensity=1.0, effective_radius=0.5
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

aplt.Tracer.figures(
    tracer=tracer,
    grid=grid,
    include=aplt.Include2D(critical_curves=True),
    plotter=plotter,
)

aplt.Tracer.subplot_tracer(
    tracer=tracer, grid=grid, include=aplt.Include2D(caustics=True), plotter=plotter
)
