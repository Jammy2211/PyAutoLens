import numpy as np
from os import path
import pytest

from autoconf import conf
import autofit as af
import autolens as al
from autolens import exc


def test__check_positions_on_instantiation():

    al.PositionsThresholder(
        positions=al.Grid2DIrregular([(1.0, 2.0), (3.0, 4.0)]),
        threshold=0.1,
        use_resampling=True,
        use_likelihood_penalty=False,
        use_likelihood_overwrite=False,
    )

    # Positions input with threshold but positions are length 1.

    with pytest.raises(exc.PositionsException):

        al.PositionsThresholder(
            positions=al.Grid2DIrregular([(1.0, 2.0)]),
            threshold=0.1,
            use_resampling=True,
            use_likelihood_penalty=False,
            use_likelihood_overwrite=False,
        )

    # No `use_` input is True.

    with pytest.raises(exc.PositionsException):

        al.PositionsThresholder(
            positions=al.Grid2DIrregular([(1.0, 2.0), (3.0, 4.0)]),
            threshold=0.1,
            use_resampling=False,
            use_likelihood_penalty=False,
            use_likelihood_overwrite=False,
        )

    # Too many `use_` inputs are True.

    with pytest.raises(exc.PositionsException):

        al.PositionsThresholder(
            positions=al.Grid2DIrregular([(1.0, 2.0), (3.0, 4.0)]),
            threshold=0.1,
            use_resampling=True,
            use_likelihood_penalty=True,
            use_likelihood_overwrite=False,
        )
