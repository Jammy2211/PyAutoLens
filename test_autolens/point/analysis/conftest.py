import pytest

import autolens as al
import autofit as af


@pytest.fixture
def grid():
    return al.Grid2D.uniform(
        shape_native=(100, 100),
        pixel_scales=0.05,
    )


@pytest.fixture
def model():
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        mass=al.mp.Isothermal,
    )
    source = af.Model(al.ps.Point)

    return af.Collection(
        lens=lens,
        source=source,
    )
