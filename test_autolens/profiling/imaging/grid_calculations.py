import time
import autolens as al

shape_2d = (400, 400)
pixel_scales = 0.05

# The grid used to profile the simulation.
grid = al.Grid.uniform(shape_2d=shape_2d, pixel_scales=0.05, sub_size=16)

"""Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens."""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalPowerLaw(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.0, 0.05)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        elliptical_comps=(0.096225, -0.055555),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

"""Use these galaxies to setup a tracer, which will generate the image for the simulated imaging dataset."""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

start = time.time()
tracer.image_from_grid(grid=grid)
diff = time.time() - start
print("Time to Compute Profile Image Using regular Grid = {}".format(diff))

# The grid used to profile the simulation.
grid = al.GridIterate.uniform(
    shape_2d=shape_2d,
    pixel_scales=0.05,
    fractional_accuracy=0.9999,
    sub_steps=[2, 4, 6, 8, 10, 12, 14, 16],
)

start = time.time()
tracer.image_from_grid(grid=grid)
diff = time.time() - start
print("Time to Compute Profile Image Using GridIterate = {}".format(diff))
