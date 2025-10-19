import logging
from dataclasses import dataclass

import autoarray as aa

logger = logging.getLogger(__name__)


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
    plane_triangles: aa.AbstractTriangles
