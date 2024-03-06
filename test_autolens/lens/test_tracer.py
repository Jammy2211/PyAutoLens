import numpy as np
import pytest
from os import path
from skimage import measure

import autofit as af
import autolens as al
from autoconf.dictable import from_json, output_to_json

### Planes ###


def test__plane_redshifts():
    g1 = al.Galaxy(redshift=1)
    g2 = al.Galaxy(redshift=2)
    g3 = al.Galaxy(redshift=3)

    tracer = al.Tracer(galaxies=[g1, g2])

    assert tracer.plane_redshifts == [1, 2]

    tracer = al.Tracer(galaxies=[g2, g2, g3, g1, g1])

    assert tracer.plane_redshifts == [1, 2, 3]


# def test__planes():
#
#     g1 = al.Galaxy(redshift=1)
#     g2 = al.Galaxy(redshift=2)
#
#     tracer = al.Tracer(galaxies=[g1, g2])
#
#     assert tracer.planes == [[g1], [g2]]


### Has Attributes ###

