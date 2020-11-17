import autolens as al
import autolens.plot as aplt

plotter = aplt.Plotter()
sub_plotter = aplt.SubPlotter()

grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.SphericalExponential(centre=(0.0, 0.0), intensity=1.0),
    light_1=al.lp.SphericalExponential(centre=(1.0, 1.0), intensity=1.0),
    light_2=al.lp.SphericalExponential(centre=(-1.0, 0.5), intensity=1.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalExponential(
        centre=(0.02, 0.01), intensity=1.0, effective_radius=0.5
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

aplt.LightProfile.image(light_profile=lens_galaxy.light, grid=grid)
aplt.Galaxy.image(galaxy=lens_galaxy, grid=grid)
aplt.Plane.image(plane=tracer.image_plane, grid=grid)

aplt.Tracer.image(
    tracer=tracer,
    grid=grid,
    include=aplt.Include(critical_curves=True),
    plotter=plotter,
)
