import autolens as al
import autolens.plot as aplt

# To begin chapter 4, we'll begin by learning about pixelizations, which we apply to a source-plane to reconstruct a
# source galaxy's light.

# Lets setup a lensed source-plane grid, using a lens galaxy and tracer (our source galaxy doesn't have a light profile,
# as we're going to reconstruct its light using a pixelization).
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=1.6
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[1]

# Next, lets set up a pixelization using the 'pixelizations' module, which we've imported as 'pix'.

# There are multiple pixelizations available in PyAutoLens. For now, we'll keep it simple and use a uniform
# rectangular grid. As usual, the grid's 'shape' defines its (y,x) dimensions.
rectangular = al.pix.Rectangular(shape=(25, 25))

# By itself, a pixelization doesn't tell us much. It has no grid of coordinates, no image, and nothing which tells it
# about the lens we're fitting. This information comes when we use the pixelization to set up a 'mapper'. We'll use
# the (traced) source-plane grid to set up this mapper.

mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)

# Infact, we can plot the source-plane grid and rectangular pixelization grid on our pixelization - to make it look
# slightly less boring!
aplt.MapperObj(
    mapper=mapper,
    include=aplt.Include(inversion_grid=True, inversion_pixelization_grid=True),
    plotter=aplt.Plotter(
        labels=aplt.Labels(title="Slightly less Boring Grid of Rectangular Pixels")
    ),
)
