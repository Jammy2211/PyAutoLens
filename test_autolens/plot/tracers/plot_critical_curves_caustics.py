import autolens as al
from autoastro.plot import lensing_plotters

plotter = aplt.Plotter()
sub_plotter = aplt.SubPlotter()

mask = al.Mask.circular(
    shape_2d=(200, 200), pixel_scales=0.03, sub_size=1, radius=2.4, centre=(0.0, 0.0)
)

grid = al.Grid.from_mask(mask=mask)

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

aplt.Tracer.profile_image(
    tracer=tracer,
    grid=grid,
    mask=mask,
    include=lensing_plotters.Include(critical_curves=True),
    plotter=plotter,
)

aplt.Tracer.subplot_tracer(
    include=lensing_plotters.Include(caustics=True), sub_plotter=sub_plotter
)
