import pytest
import autolens as al


@pytest.fixture
def grid():
    return al.Grid2D.uniform(
        shape_native=(100, 100),
        pixel_scales=0.05,
    )
