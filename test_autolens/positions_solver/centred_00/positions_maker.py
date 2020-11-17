# %%
"""
__Positions Maker__

This tool creates a set of positions using the _peaks criteria from a high resolution grid, without any buffering or
upscaling. This means:

    - It will not incorrectly remove any true multiple images due to grid buffering / refinement.
    - Extra multiple images will be includded corresponding to peaks in the mass profile that are local, not global.

These results are used to test whether more efficient position solvers implementations lose multiple images.
"""

# %%
import autofit as af
import autolens as al
import autolens.plot as aplt
import os

# %%
"""The pickle path is where the `Tracer` and `Positions` are output, so they can be loaded by other scripts."""

# %%
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))
pickle_path = f"{path}/pickles"

# %%
"""A high resolution grid is used to ensure positions are computed to a given accuracy."""

# %%
grid = al.Grid.uniform(
    shape_2d=(600, 600),
    pixel_scales=0.01,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

# %%
"""
The mass-profile and source light profile in this example have fixed centre (0.0, 0.0), restricting the range of 
lensing geometries.
"""

# %%
mass_profile_model = af.PriorModel(al.mp.EllipticalIsothermal)
mass_profile_model.centre.centre_0 = 0.0
mass_profile_model.centre.centre_1 = 0.0
mass_profile_model.elliptical_comps.ellipitical_comps_0 = af.UniformPrior(
    lower_limit=-1.0, upper_limit=1.0
)
mass_profile_model.elliptical_comps.ellipitical_comps_1 = af.UniformPrior(
    lower_limit=-1.0, upper_limit=1.0
)
mass_profile_model.centre.einstein_radius = af.UniformPrior(
    lower_limit=0.3, upper_limit=2.0
)

iters = 50

"""Use a `PositionsSolver` which does not use grid upscaling."""

solver = al.PositionsFinder(grid=grid, use_upscaling=True, upscale_factor=2)

for i in range(iters):

    """Make a random `MassProfile` instance from the priors defined above."""

    mass_profile = mass_profile_model.random_instance()

    """
    Only the `LightProfile` centre is used by the position solver, but a light profile is used to visalize the
    lensed source.
    """

    exponential_light_profile = al.lp.EllipticalExponential(
        centre=(0.0, 0.0),
        elliptical_comps=(0.2, 0.0),
        intensity=0.05,
        effective_radius=0.2,
    )

    """Setup the lens, source and _Tracer_."""

    lens_galaxy = al.Galaxy(redshift=0.5, mass=mass_profile)
    source_galaxy = al.Galaxy(redshift=1.0, light=exponential_light_profile)
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    """Solve for the positions via the _Tracer_."""

    positions = solver.solve(
        lensing_obj=tracer,
        source_plane_coordinate=tracer.source_plane.galaxies[0].light.centre,
    )

    """Visually inspect the positions (comment this out if you are confident the code is behaving as expected)."""

    aplt.Tracer.image(
        tracer=tracer,
        grid=grid,
        positions=positions,
        include=aplt.Include(origin=False, critical_curves=False, caustics=False),
    )

    """Save the `Tracer` and `Positions` so they can be used for testing other `PositionsSolver` settings."""

    tracer.save(file_path=pickle_path, filename=f"tracer_{str(i)}")
    positions.save(file_path=pickle_path, filename=f"positions_{str(i)}")
