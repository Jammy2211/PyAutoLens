from typing import Optional, List

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autogalaxy.aggregator.interferometer import _interferometer_from
from autogalaxy.aggregator.abstract import AbstractAgg

from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.analysis.preloads import Preloads

from autolens.aggregator.tracer import _tracer_from


def _fit_interferometer_from(
    fit: af.Fit,
    galaxies: List[ag.Galaxy],
    real_space_mask: Optional[aa.Mask2D] = None,
    settings_interferometer: aa.SettingsInterferometer = None,
    settings_pixelization: aa.SettingsPixelization = None,
    settings_inversion: aa.SettingsInversion = None,
    use_preloaded_grid: bool = True,
) -> FitInterferometer:
    """
    Returns a `FitInterferometer` object from a PyAutoFit database `Fit` object and an instance of galaxies from a non-linear
    search model-fit.

    This function adds the `hyper_model_image` and `hyper_galaxy_image_path_dict` to the galaxies before performing the
    fit, if they were used.

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of PyAutoGalaxy model-fits.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.

    Returns
    -------
    FitInterferometer
        The fit to the interferometer dataset computed via an instance of galaxies.
    """
    interferometer = _interferometer_from(
        fit=fit,
        real_space_mask=real_space_mask,
        settings_interferometer=settings_interferometer,
    )
    tracer = _tracer_from(fit=fit, galaxies=galaxies)

    settings_pixelization = settings_pixelization or fit.value(
        name="settings_pixelization"
    )
    settings_inversion = settings_inversion or fit.value(name="settings_inversion")

    preloads = None

    if use_preloaded_grid:

        sparse_grids_of_planes = fit.value(name="preload_sparse_grids_of_planes")

        if sparse_grids_of_planes is not None:

            preloads = Preloads(sparse_image_plane_grid_pg_list=sparse_grids_of_planes)

    return FitInterferometer(
        dataset=interferometer,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
        preloads=preloads,
    )


class FitInterferometerAgg(AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        settings_interferometer: Optional[aa.SettingsInterferometer] = None,
        settings_pixelization: Optional[aa.SettingsPixelization] = None,
        settings_inversion: Optional[aa.SettingsInversion] = None,
        use_preloaded_grid: bool = True,
        real_space_mask: Optional[aa.Mask2D] = None,
    ):
        """
        Wraps a PyAutoFit aggregator in order to create generators of fits to interferometer data, corresponding to the
        results of a non-linear search model-fit.
        """
        super().__init__(aggregator=aggregator)

        self.settings_interferometer = settings_interferometer
        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion
        self.use_preloaded_grid = use_preloaded_grid
        self.real_space_mask = real_space_mask

    def make_object_for_gen(self, fit, galaxies) -> FitInterferometer:
        """
        Creates a `FitInterferometer` object from a `ModelInstance` that contains the galaxies of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of PyAutoGalaxy model-fits.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.

        Returns
        -------
        FitInterferometer
            A fit to interferometer data whose galaxies are a sample of a PyAutoFit non-linear search.
        """
        return _fit_interferometer_from(
            fit=fit,
            galaxies=galaxies,
            settings_interferometer=self.settings_interferometer,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            use_preloaded_grid=self.use_preloaded_grid,
        )
