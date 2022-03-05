import numpy as np
import pytest

import autolens as al


def test__fit_via_mock_profile(dataset_quantity_7x7_array_2d):

    model_object = al.m.MockMassProfile(
        convergence_2d=al.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0),
        potential_2d=al.Array2D.full(
            fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0
        ),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[al.Galaxy(redshift=0.5, mass=model_object)]
    )

    fit_quantity = al.FitQuantity(
        dataset=dataset_quantity_7x7_array_2d,
        tracer=tracer,
        func_str="convergence_2d_from",
    )

    assert fit_quantity.chi_squared == pytest.approx(0.0, 1.0e-4)

    assert fit_quantity.log_likelihood == pytest.approx(
        -0.5 * 49.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
    )

    fit_quantity = al.FitQuantity(
        dataset=dataset_quantity_7x7_array_2d,
        tracer=tracer,
        func_str="potential_2d_from",
    )

    assert fit_quantity.chi_squared == pytest.approx(12.25, 1.0e-4)

    assert fit_quantity.log_likelihood == pytest.approx(-85.1171999, 1.0e-4)
