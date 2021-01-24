from os import path
import autolens as al
import autolens.plot as aplt

from test_autogalaxy.simulators.imaging import instrument_util

test_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "..", "..")


def pixel_scale_from_instrument(instrument):
    """
    Returns the pixel scale from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """
    if instrument in "vro":
        return (0.2, 0.2)
    elif instrument in "euclid":
        return (0.1, 0.1)
    elif instrument in "hst":
        return (0.05, 0.05)
    elif instrument in "hst_up":
        return (0.03, 0.03)
    elif instrument in "ao":
        return (0.01, 0.01)
    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def grid_from_instrument(instrument):
    """
    Returns the `Grid2D` from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """
    if instrument in "vro":
        return al.Grid2DIterate.uniform(shape_native=(80, 80), pixel_scales=0.2)
    elif instrument in "euclid":
        return al.Grid2DIterate.uniform(shape_native=(120, 120), pixel_scales=0.1)
    elif instrument in "hst":
        return al.Grid2DIterate.uniform(shape_native=(200, 200), pixel_scales=0.05)
    elif instrument in "hst_up":
        return al.Grid2DIterate.uniform(shape_native=(300, 300), pixel_scales=0.03)
    elif instrument in "ao":
        return al.Grid2DIterate.uniform(shape_native=(800, 800), pixel_scales=0.01)
    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def psf_from_instrument(instrument):
    """
    Returns the *PSF* from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """
    if instrument in "vro":
        return al.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.5, pixel_scales=0.2, renormalize=True
        )

    elif instrument in "euclid":
        return al.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.1, pixel_scales=0.1, renormalize=True
        )
    elif instrument in "hst":
        return al.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.05, pixel_scales=0.05, renormalize=True
        )
    elif instrument in "hst_up":
        return al.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.05, pixel_scales=0.03, renormalize=True
        )
    elif instrument in "ao":
        return al.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.025, pixel_scales=0.01, renormalize=True
        )

    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def simulator_from_instrument(instrument):
    """
    Returns the *Simulator* from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """

    grid = grid_from_instrument(instrument=instrument)
    psf = psf_from_instrument(instrument=instrument)

    if instrument in "vro":
        return al.SimulatorImaging(
            exposure_time=100.0,
            psf=psf,
            background_sky_level=1.0,
            add_poisson_noise=True,
        )
    elif instrument in "euclid":
        return al.SimulatorImaging(
            exposure_time=2260.0,
            psf=psf,
            background_sky_level=1.0,
            add_poisson_noise=True,
        )
    elif instrument in "hst":
        return al.SimulatorImaging(
            exposure_time=2000.0,
            psf=psf,
            background_sky_level=1.0,
            add_poisson_noise=True,
        )
    elif instrument in "hst_up":
        return al.SimulatorImaging(
            exposure_time=2000.0,
            psf=psf,
            background_sky_level=1.0,
            add_poisson_noise=True,
        )
    elif instrument in "ao":
        return al.SimulatorImaging(
            exposure_time=1000.0,
            psf=psf,
            background_sky_level=1.0,
            add_poisson_noise=True,
        )
    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def simulate_imaging_from_instrument(instrument, dataset_name, galaxies):

    # Simulate the imaging data, remembering that we use a special image which ensures edge-effects don't
    # degrade our modeling of the telescope optics (e.al. the PSF convolution).

    grid = instrument_util.grid_from_instrument(instrument=instrument)

    simulator = simulator_from_instrument(instrument=instrument)

    # Use the input galaxies to setup a tracer, which will generate the image for the simulated imaging data.
    tracer = al.Tracer.from_galaxies(galaxies=galaxies)

    imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

    # Now, lets output this simulated imaging-data to the test_autoarray/simulator folder.
    test_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "..", ".."
    )

    dataset_path = path.join(test_path, "dataset", "imaging", dataset_name, instrument)

    imaging.output_to_fits(
        image_path=path.join(dataset_path, "image.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        overwrite=True,
    )

    plotter = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))
    plotter = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

    aplt.Imaging.subplot_imaging(imaging=imaging, plotter=plotter)

    aplt.imaging.individual(
        imaging=imaging,
        image=True,
        noise_map=True,
        psf=True,
        signal_to_noise_map=True,
        plotter=plotter,
    )

    aplt.Tracer.subplot_tracer(tracer=tracer, grid=grid, plotter=plotter)

    aplt.Tracer.figures(
        tracer=tracer,
        grid=grid,
        image=True,
        source_plane=True,
        convergence=True,
        potential=True,
        deflections=True,
        plotter=plotter,
    )


def load_test_imaging(dataset_name, instrument, name=None):

    pixel_scales = instrument_util.pixel_scale_from_instrument(instrument=instrument)

    test_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "..", ".."
    )

    dataset_path = path.join(test_path, "dataset", "imaging", dataset_name, instrument)

    return al.Imaging.from_fits(
        image_path=path.join(dataset_path, "image.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=pixel_scales,
        name=name,
    )
