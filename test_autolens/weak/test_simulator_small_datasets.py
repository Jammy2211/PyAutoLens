import pytest

import autolens as al


def _tracer():
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6),
    )
    return al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])


def test__random_positions__capped_under_small_datasets(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    dataset = al.SimulatorShearYX(noise_sigma=0.3, seed=1).via_tracer_random_positions_from(
        tracer=_tracer(), n_galaxies=200, grid_extent=3.0
    )

    assert dataset.n_galaxies == 25


def test__random_positions__uncapped_without_env_var(monkeypatch):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    dataset = al.SimulatorShearYX(noise_sigma=0.3, seed=1).via_tracer_random_positions_from(
        tracer=_tracer(), n_galaxies=40, grid_extent=3.0
    )

    assert dataset.n_galaxies == 40


def test__explicit_grid__never_capped(monkeypatch):
    import autoarray as aa
    import numpy as np

    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    grid = aa.Grid2DIrregular(values=np.random.default_rng(1).uniform(-3, 3, (60, 2)))
    dataset = al.SimulatorShearYX(noise_sigma=0.3, seed=1).via_tracer_from(
        tracer=_tracer(), grid=grid
    )

    assert dataset.n_galaxies == 60


def _solver():
    import autolens as al

    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)
    return al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    )


def _tracer_with_einstein_radius(einstein_radius):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=einstein_radius),
    )
    return al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])


def test__point_solver__short_circuits_to_a_model_independent_pair(monkeypatch):
    """Under the cap the solve is skipped entirely, so every lens model yields the same
    two positions. Anything derived from them is model-independent — the reason a pinned
    parity literal cannot be compared against a capped run (PyAutoLens#710)."""
    import numpy as np

    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    solver = _solver()

    solved = [
        np.asarray(
            solver.solve(
                tracer=_tracer_with_einstein_radius(einstein_radius),
                source_plane_coordinate=(0.07, 0.07),
            ).array
        )
        for einstein_radius in (1.0, 1.6, 2.5)
    ]

    for positions in solved:
        assert positions == pytest.approx(np.array([[1.0, 0.0], [0.0, 1.0]]))


def test__point_solver__short_circuit_warns_once_per_process(monkeypatch, caplog):
    """The short-circuit must not be silent, and must not flood a vmap batch."""
    import logging

    from autolens.point.solver import point_solver

    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")
    monkeypatch.setattr(point_solver, "_SMALL_DATASETS_WARNED", False)

    solver = _solver()
    tracer = _tracer_with_einstein_radius(1.6)

    with caplog.at_level(logging.WARNING, logger=point_solver.__name__):
        for _ in range(3):
            solver.solve(tracer=tracer, source_plane_coordinate=(0.07, 0.07))

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "PYAUTO_SMALL_DATASETS" in record.getMessage()
    ]

    assert len(warnings) == 1
    assert "EVERY lens model" in warnings[0].getMessage()


def test__point_solver__no_short_circuit_warning_without_the_env_var(monkeypatch, caplog):
    import logging

    from autolens.point.solver import point_solver

    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    monkeypatch.setattr(point_solver, "_SMALL_DATASETS_WARNED", False)

    with caplog.at_level(logging.WARNING, logger=point_solver.__name__):
        _solver().solve(
            tracer=_tracer_with_einstein_radius(1.6),
            source_plane_coordinate=(0.07, 0.07),
        )

    assert not [
        record
        for record in caplog.records
        if "PYAUTO_SMALL_DATASETS" in record.getMessage()
    ]
