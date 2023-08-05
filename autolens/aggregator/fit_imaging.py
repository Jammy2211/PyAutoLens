from typing import Optional, List

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autogalaxy.aggregator.imaging import _imaging_from
from autogalaxy.aggregator.abstract import AbstractAgg

from autolens.imaging.fit_imaging import FitImaging
from autolens.analysis.preloads import Preloads
from autolens.aggregator.tracer import _tracer_from


def _fit_imaging_from(
    fit: af.Fit,
    galaxies: List[ag.Galaxy],
    settings_dataset: aa.SettingsImaging = None,
    settings_pixelization: aa.SettingsPixelization = None,
    settings_inversion: aa.SettingsInversion = None,
    use_preloaded_grid: bool = True,
) -> FitImaging:
    """
    Returns an `FitImaging` object from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The imaging data, noise-map, PSF and settings as .fits files (e.g. `dataset/data.fits`).
    - The mask used to mask the `Imaging` data structure in the fit (`dataset/mask.fits`).
    - The settings of pixelization used by the fit (`dataset/settings_pixelization.json`).
    - The settings of inversions used by the fit (`dataset/settings_inversion.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `FitImaging` object for a given non-linear search sample
    (e.g. the maximum likelihood model). This includes associating adapt images with their respective galaxies.

    The settings of a pixelization of inversion can be overwritten by inputting a `settings_dataset` object, for example
    if you want to use a grid with a different inversion solver.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.
    settings_dataset
        Optionally overwrite the `SettingsImaging` of the `Imaging` object that is created from the fit.
    settings_pixelization
        Optionally overwrite the `SettingsPixelization` of the `Pixelization` object that is created from the fit.
    settings_inversion
        Optionally overwrite the `SettingsInversion` of the `Inversion` object that is created from the fit.
    use_preloaded_grid
        Certain pixelization's construct their mesh in the source-plane from a stochastic KMeans algorithm. This grid
        may be output to hard-disk after the model-fit and loaded via the database to ensure the same grid is used
        as the fit.
    """

    dataset = _imaging_from(fit=fit, settings_dataset=settings_dataset)

    tracer = _tracer_from(fit=fit, galaxies=galaxies)

    settings_pixelization = settings_pixelization or fit.value(
        name="settings_pixelization"
    )
    settings_inversion = settings_inversion or fit.value(name="settings_inversion")

    preloads = Preloads(use_w_tilde=False)

    if use_preloaded_grid:
        sparse_grids_of_planes = fit.value(name="preload_sparse_grids_of_planes")

        if sparse_grids_of_planes is not None:
            preloads = Preloads(
                sparse_image_plane_grid_pg_list=sparse_grids_of_planes,
                use_w_tilde=False,
            )

            if len(preloads.sparse_image_plane_grid_pg_list) == 2:
                if type(preloads.sparse_image_plane_grid_pg_list[1]) != list:
                    preloads.sparse_image_plane_grid_pg_list[1] = [
                        preloads.sparse_image_plane_grid_pg_list[1]
                    ]

    return FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
        preloads=preloads,
    )


class FitImagingAgg(AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        settings_dataset: Optional[aa.SettingsImaging] = None,
        settings_pixelization: Optional[aa.SettingsPixelization] = None,
        settings_inversion: Optional[aa.SettingsInversion] = None,
        use_preloaded_grid: bool = True,
    ):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `FitImaging` objects from the results
        of a model-fit.

        The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

        - The imaging data, noise-map, PSF and settings as .fits files (e.g. `dataset/data.fits`).
        - The mask used to mask the `Imaging` data structure in the fit (`dataset/mask.fits`).
        - The settings of pixelization used by the fit (`dataset/settings_pixelization.json`).
        - The settings of inversions used by the fit (`dataset/settings_inversion.json`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `FitImaging` object via the `_fit_imaging_from` method.

        This class's methods returns generators which create the instances of the `FitImaging` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `FitImaging` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `FitImaging` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        settings_dataset
            Optionally overwrite the `SettingsImaging` of the `Imaging` object that is created from the fit.
        settings_pixelization
            Optionally overwrite the `SettingsPixelization` of the `Pixelization` object that is created from the fit.
        settings_inversion
            Optionally overwrite the `SettingsInversion` of the `Inversion` object that is created from the fit.
        use_preloaded_grid
            Certain pixelization's construct their mesh in the source-plane from a stochastic KMeans algorithm. This
            grid may be output to hard-disk after the model-fit and loaded via the database to ensure the same grid is
            used as the fit.
        """
        super().__init__(aggregator=aggregator)

        self.settings_dataset = settings_dataset
        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion
        self.use_preloaded_grid = use_preloaded_grid

    def object_via_gen_from(self, fit, galaxies) -> FitImaging:
        """
        Returns a generator of `FitImaging` objects from an input aggregator.

        See `__init__` for a description of how the `FitImaging` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.
        """
        return _fit_imaging_from(
            fit=fit,
            galaxies=galaxies,
            settings_dataset=self.settings_dataset,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            use_preloaded_grid=self.use_preloaded_grid,
        )
