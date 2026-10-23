import numpy as np
import pytest

import autofit as af
import autolens as al

from autolens.point.model.result import ResultPoint


def _result_point(dataset_name="point_0"):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), einstein_radius=1.0, ell_comps=(0.0, 0.1)
        ),
    )
    source = al.Galaxy(
        redshift=1.0, point_0=al.ps.PointFlux(centre=(0.0, 0.0), flux=1.0)
    )

    instance = af.ModelInstance()
    instance.galaxies = af.ModelInstance()
    instance.galaxies.lens = lens
    instance.galaxies.source = source

    positions = al.Grid2DIrregular([(0.0, 1.8), (0.0, -0.6)])

    dataset = al.PointDataset(
        name=dataset_name,
        positions=positions,
        positions_noise_map=al.ArrayIrregular([0.05, 0.05]),
        fluxes=al.ArrayIrregular([1.0, 1.0]),
        fluxes_noise_map=al.ArrayIrregular([0.1, 0.1]),
    )

    solver = al.m.MockPointSolver(model_positions=positions)

    analysis = al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=False)

    samples_summary = al.m.MockSamplesSummary(max_log_likelihood_instance=instance)

    return ResultPoint(samples_summary=samples_summary, analysis=analysis)


def test__magnification_accessors__delegate_to_max_log_likelihood_fit():
    result = _result_point()

    fit = result.max_log_likelihood_fit

    signed = result.signed_magnifications_at_positions
    unsigned = result.magnifications_at_positions

    assert isinstance(signed, al.ArrayIrregular)
    assert isinstance(unsigned, al.ArrayIrregular)

    assert signed.array == pytest.approx(
        fit.positions.signed_magnifications_at_positions.array, 1.0e-10
    )
    assert signed.array == pytest.approx(
        fit.flux.signed_magnifications_at_positions.array, 1.0e-10
    )
    assert signed.array[0] > 0.0
    assert signed.array[1] < 0.0
    assert (np.asarray(unsigned.array) == np.abs(np.asarray(signed.array))).all()


def test__magnification_accessors__no_component_fit__raises_naming_fit_type():
    result = _result_point(dataset_name="point_unpaired")

    with pytest.raises(AttributeError, match="FitPointDataset"):
        result.signed_magnifications_at_positions

    with pytest.raises(AttributeError, match="FitPointDataset"):
        result.magnifications_at_positions
