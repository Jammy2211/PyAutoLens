from os import path
import autolens as al
import autolens.plot as aplt

from test_autogalaxy.simulators.interferometer import instrument_util

test_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "..", "..")


def simulator_from_instrument(instrument):
    """
    Returns the pixel scale from an instrument based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """

    uv_wavelengths = instrument_util.uv_wavelengths_from_instrument(
        instrument=instrument
    )
    grid = instrument_util.grid_from_instrument(instrument=instrument)

    if instrument in "sma":
        return al.SimulatorInterferometer(
            uv_wavelengths=uv_wavelengths,
            exposure_time=100.0,
            background_sky_level=1.0,
            noise_sigma=0.01,
        )
    else:
        raise ValueError(
            "An invalid dataset_name resolution was entered - ", instrument
        )


def simulate_interferometer_from_instrument(instrument, dataset_name, galaxies):

    # Simulate the imaging data, remembering that we use a special image which ensures edge-effects don't
    # degrade our modeling of the telescope optics (e.al. the PSF convolution).

    grid = instrument_util.grid_from_instrument(instrument=instrument)

    simulator = simulator_from_instrument(instrument=instrument)

    # Use the input galaxies to setup a tracer, which will generate the image for the simulated imaging data.
    tracer = al.Tracer.from_galaxies(galaxies=galaxies)

    interferometer = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

    # Now, lets output this simulated interferometer-simulator to the test_autoarray/simulator folder.
    test_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "..", ".."
    )

    dataset_path = path.join("dataset", "interferometer", dataset_name, instrument)

    interferometer.output_to_fits(
        visibilities_path=path.join(dataset_path, "visibilities.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
        overwrite=True,
    )

    plotter = aplt.Plotter(output=aplt.Output(path=dataset_path, format="png"))
    sub_plotter = aplt.SubPlotter(output=aplt.Output(path=dataset_path, format="png"))

    aplt.Interferometer.subplot_interferometer(
        interferometer=interferometer, sub_plotter=sub_plotter
    )

    aplt.Interferometer.individual(
        interferometer=interferometer, plot_visibilities=True, plotter=plotter
    )

    aplt.Tracer.subplot_tracer(tracer=tracer, grid=grid, sub_plotter=sub_plotter)

    aplt.Tracer.individual(
        tracer=tracer,
        grid=grid,
        plot_image=True,
        plot_source_plane=True,
        plot_convergence=True,
        plot_potential=True,
        plot_deflections=True,
        plotter=plotter,
    )


def load_test_interferometer(dataset_name, instrument):

    dataset_path = path.join("dataset", "interferometer", dataset_name, instrument)

    return al.Interferometer.from_fits(
        visibilities_path=path.join(dataset_path, "visibilities.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    )
