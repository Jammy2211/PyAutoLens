# %%
"""
__Positions Tester__

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
from test_autolens.positions_solver import util
import os

# %%
"""The pickle path is where the `Tracer` and `Positions` are output, so they can be loaded by other scripts."""

# %%
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))
pickle_path = f"{path}/pickles"

# %%
"""The initial grid for position solver which is upscaled iteratively by the solver."""

# %%
grid = al.Grid.uniform(
    shape_2d=(2000, 2000),
    pixel_scales=0.005,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

"""Use a `PositionsSolver` which uses grid upscaling."""

solver = al.PositionsFinder(
    grid=grid,
    use_upscaling=True,
    pixel_scale_precision=0.0000001,
    upscale_factor=2,
    distance_from_source_centre=0.1,
)

iters = 50

for i in range(40, 41):

    tracer = al.Tracer.load(file_path=pickle_path, filename=f"tracer_{str(i)}")

    positions = solver.solve(
        lensing_obj=tracer,
        source_plane_coordinate=tracer.source_plane.galaxies[0].light.centre,
    )

    positions_true = al.GridCoordinates.load(
        file_path=pickle_path, filename=f"positions_{str(i)}"
    )

    if positions is not None:
        minimum_separations = util.minimum_separations_from(
            positions_true=positions_true, positions=positions
        )

        in_positions_true = util.check_if_positions_in_positions_true(
            positions_true=positions_true, positions=positions, threshold=0.1
        )

        positions_plot = al.GridCoordinates(
            coordinates=[positions.in_list[0], positions_true.in_list[0]]
        )

    else:

        minimum_separations = None
        in_positions_true = None

        positions_plot = al.GridCoordinates(coordinates=[positions_true.in_list[0]])

    print(positions)

    aplt.Tracer.image(
        tracer=tracer,
        grid=grid,
        positions=positions_plot,
        include=aplt.Include(origin=False, critical_curves=True, caustics=False),
    )
