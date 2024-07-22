import pytest
import autolens as al


@pytest.fixture
def grid():
    return al.Grid2D.uniform(
        shape_native=(10, 10),
        pixel_scales=1.0,
    )


@pytest.fixture
def tracer():
    isothermal_mass_profile = al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=isothermal_mass_profile,
    )

    point_source = al.ps.PointSourceChi(centre=(0.07, 0.07))

    source_galaxy = al.Galaxy(redshift=1.0, point_0=point_source)

    return al.Tracer(galaxies=[lens_galaxy, source_galaxy])
