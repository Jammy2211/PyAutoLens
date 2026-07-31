from pathlib import Path

import autoarray as aa
import autolens as al

import pytest

from autolens.weak.model.plotter import PlotterWeak

directory = Path(__file__).resolve().parent


def _isothermal_tracer(einstein_radius=1.6, ell_comps=(0.0, 0.05)):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
        ),
    )
    source = al.Galaxy(redshift=1.0)
    return al.Tracer(galaxies=[lens, source])


@pytest.fixture(name="fit_weak")
def make_fit_weak():
    grid = aa.Grid2DIrregular(
        values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6), (-1.1, -0.8)]
    )
    truth = _isothermal_tracer(einstein_radius=1.6)
    dataset = al.SimulatorShearYX(noise_sigma=0.0, seed=0).via_tracer_from(
        tracer=truth, grid=grid, name="test"
    )
    dataset.noise_map = aa.ArrayIrregular(values=[0.3, 0.3, 0.3, 0.3])
    model = _isothermal_tracer(einstein_radius=1.5)
    return al.FitWeak(dataset=dataset, tracer=model)


@pytest.fixture(name="plot_path")
def make_plot_path():
    return directory / "files"


def test__fit_weak__quick_update__writes_normal_fit_subplot(
    fit_weak, plot_path, plot_patch
):
    plotter = PlotterWeak(image_path=plot_path)

    plotter.fit_weak(fit=fit_weak, quick_update=True)

    assert str(plot_path / "subplot_fit_weak.png") in plot_patch.paths
    assert str(plot_path / "fit_quick.png") not in plot_patch.paths
