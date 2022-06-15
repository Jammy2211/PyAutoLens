import numpy as np
from os import path
import pytest

from autoconf import conf
import autofit as af
import autolens as al
from autolens import exc


def test__check_positions_on_instantiation():

    al.PositionsResample(
        positions=al.Grid2DIrregular([(1.0, 2.0), (3.0, 4.0)]),
        threshold=0.1,
    )

    # Positions input with threshold but positions are length 1.

    with pytest.raises(exc.PositionsException):

        al.PositionsResample(
            positions=al.Grid2DIrregular([(1.0, 2.0)]),
            threshold=0.1,
        )


