import os
from os import path
import pytest

from autofit.tools.util import open_

import autolens as al
from autolens import exc


def test__check_positions_on_instantiation():
    al.PositionsLHResample(
        positions=al.Grid2DIrregular([(1.0, 2.0), (3.0, 4.0)]), threshold=0.1
    )

    # Positions input with threshold but positions are length 1.

    with pytest.raises(exc.PositionsException):
        al.PositionsLHResample(
            positions=al.Grid2DIrregular([(1.0, 2.0)]), threshold=0.1
        )


def test__output_positions_info():
    output_path = path.join(
        "{}".format(os.path.dirname(os.path.realpath(__file__))), "files"
    )

    positions_likelihood = al.PositionsLHResample(
        positions=al.Grid2DIrregular([(1.0, 2.0), (3.0, 4.0)]), threshold=0.1
    )

    tracer = al.m.MockTracer(
        traced_grid_2d_list_from=al.Grid2DIrregular(values=[[(0.5, 1.5), (2.5, 3.5)]])
    )

    positions_likelihood.output_positions_info(output_path=output_path, tracer=tracer)

    positions_file = path.join(output_path, "positions.info")

    with open_(positions_file, "r") as f:
        output = f.readlines()

    assert "Positions" in output[0]

    os.remove(positions_file)
