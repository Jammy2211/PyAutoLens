import logging
from dataclasses import dataclass

import autoarray as aa
from autoarray.numpy_wrapper import register_pytree_node_class

try:
    from autoarray.structures.triangles.jax_array import ArrayTriangles
except ImportError:
    from autoarray.structures.triangles.array import ArrayTriangles


logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclass
class Step:
    """
    A step in the triangle solver algorithm.

    Attributes
    ----------
    number
        The number of the step.
    initial_triangles
        The triangles at the start of the step.
    filtered_triangles
        The triangles trace to triangles that contain the source plane coordinate.
    neighbourhood
        The neighbourhood of the filtered triangles.
    up_sampled
        The neighbourhood up-sampled to increase the resolution.
    """

    number: int
    initial_triangles: aa.AbstractTriangles
    filtered_triangles: aa.AbstractTriangles
    neighbourhood: aa.AbstractTriangles
    up_sampled: aa.AbstractTriangles
    source_triangles: aa.AbstractTriangles

    def tree_flatten(self):
        return (
            self.number,
            self.initial_triangles,
            self.filtered_triangles,
            self.neighbourhood,
            self.up_sampled,
        ), ()

    @classmethod
    def tree_unflatten(cls, _, values):
        return cls(*values)
