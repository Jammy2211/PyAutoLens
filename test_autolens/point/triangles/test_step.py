import numpy as np

import autolens as al
from autoarray.structures.triangles.shape import Point
from autolens.point.solver.step import Step


def test__step_tree_flatten_round_trip(tracer, grid):
    """
    `Step.tree_flatten` returns every field, including `plane_triangles`, so `tree_unflatten` rebuilds the step
    (it previously returned 5 children for 6 fields and raised `TypeError`).
    """
    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.25, magnification_threshold=0.1
    )
    step = next(iter(solver.steps(tracer=tracer, shape=Point(0.07, 0.07), xp=np)))

    children, aux = step.tree_flatten()
    assert len(children) == 6

    rebuilt = Step.tree_unflatten(aux, children)

    assert rebuilt.number == step.number
    assert rebuilt.plane_triangles is step.plane_triangles
    assert rebuilt.up_sampled is step.up_sampled
