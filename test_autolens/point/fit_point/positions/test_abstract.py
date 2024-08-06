import pytest

import autolens as al

def test__multi_plane_position_solving():
    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

    g0 = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))
    g1 = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.1, 0.1)))
    g2 = al.Galaxy(redshift=2.0, point_1=al.ps.Point(centre=(0.1, 0.1)))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    dataset = al.Grid2DIrregular([(0.0, 0.0), (3.0, 4.0)])
    noise_map = al.ArrayIrregular([0.5, 1.0])

    solver = al.PointSolver.for_grid(grid=grid, pixel_scale_precision=0.01)

    fit_0 = al.AbstractFitPositionsImagePair(
        name="point_0",
        data=dataset,
        noise_map=noise_map,
        tracer=tracer,
        solver=solver,
    )

    fit_1 = al.AbstractFitPositionsImagePair(
        name="point_1",
        data=dataset,
        noise_map=noise_map,  #
        tracer=tracer,
        solver=solver,
    )

    scaling_factor = tracer.cosmology.scaling_factor_between_redshifts_from(
        redshift_0=0.5, redshift_1=1.0, redshift_final=2.0
    )
    
    assert fit_0.model_data[0, 0] == pytest.approx(
        scaling_factor * fit_1.model_data[0, 0], 1.0e-1
    )
    assert fit_0.model_data[0, 1] == pytest.approx(
        scaling_factor * fit_1.model_data[0, 1], 1.0e-1
    )
