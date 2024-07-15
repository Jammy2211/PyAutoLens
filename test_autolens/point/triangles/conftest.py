import pytest
import autolens as al


@pytest.fixture
def grid():
    return al.Grid2D.uniform(
        shape_native=(10, 10),
        pixel_scales=1.0,
    )
