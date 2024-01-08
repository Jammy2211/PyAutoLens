from typing import Optional, List

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autogalaxy.aggregator.abstract import AbstractAgg

from autolens.imaging.fit_imaging import FitImaging
from autolens.analysis.preloads import Preloads

from autogalaxy.aggregator.imaging import _imaging_from
from autogalaxy.aggregator import agg_util

from autolens.aggregator.tracer import _tracer_from


def _fit_imaging_from(
    fit: af.Fit,
    galaxies: List[ag.Galaxy],
    settings_dataset: aa.SettingsImaging = None,
    settings_inversion: aa.SettingsInversion = None,
    use_preloaded_grid: bool = True,
) -> List[FitImaging]:
    """
    Returns a list of `FitImaging` object from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The imaging data, noise-map, PSF and settings as .fits files (e.g. `dataset/data.fits`).
    - The mask used to mask the `Imaging` data structure in the fit (`dataset/mask.fits`).
    - The settings of inversions used by the fit (`dataset/settings_inversion.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `FitImaging` object for a given non-linear search sample
    (e.g. the maximum likelihood model). This includes associating adapt images with their respective galaxies.

    If multiple `FitImaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
    `FitImaging` objects.

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
    settings_inversion
        Optionally overwrite the `SettingsInversion` of the `Inversion` object that is created from the fit.
    use_preloaded_grid
        Certain pixelization's construct their mesh in the source-plane from a stochastic KMeans algorithm. This grid
        may be output to hard-disk after the model-fit and loaded via the database to ensure the same grid is used
        as the fit.
    """

    dataset_list = _imaging_from(fit=fit, settings_dataset=settings_dataset)

    tracer_list = _tracer_from(fit=fit, galaxies=galaxies)

    adapt_images_list = agg_util.adapt_images_from(fit=fit)

    settings_inversion = settings_inversion or fit.value(name="settings_inversion")

    mesh_grids_of_planes_list = agg_util.mesh_grids_of_planes_list_from(
        fit=fit, total_fits=len(dataset_list), use_preloaded_grid=use_preloaded_grid
    )

    fit_dataset_list = []

    for dataset, tracer, adapt_images, mesh_grids_of_planes in zip(
        dataset_list, tracer_list, adapt_images_list, mesh_grids_of_planes_list
    ):
        preloads = agg_util.preloads_from(
            preloads_cls=Preloads,
            use_preloaded_grid=use_preloaded_grid,
            mesh_grids_of_planes=mesh_grids_of_planes,
            use_w_tilde=False,
        )

        fit_dataset_list.append(
            FitImaging(
                dataset=dataset,
                tracer=tracer,
                adapt_images=adapt_images,
                settings_inversion=settings_inversion,
                preloads=preloads,
            )
        )

    return fit_dataset_list


class FitImagingAgg(AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        settings_dataset: Optional[aa.SettingsImaging] = None,
        settings_inversion: Optional[aa.SettingsInversion] = None,
        use_preloaded_grid: bool = True,
    ):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `FitImaging` objects from the results
        of a model-fit.

        The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

        - The imaging data, noise-map, PSF and settings as .fits files (e.g. `dataset/data.fits`).
        - The mask used to mask the `Imaging` data structure in the fit (`dataset/mask.fits`).
        - The settings of inversions used by the fit (`dataset/settings_inversion.json`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `FitImaging` object via the `_fit_imaging_from` method.

        This class's methods returns generators which create the instances of the `FitImaging` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `FitImaging` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `FitImaging` objects.

        If multiple `FitImaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
        is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
        `FitImaging` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        settings_dataset
            Optionally overwrite the `SettingsImaging` of the `Imaging` object that is created from the fit.
        settings_inversion
            Optionally overwrite the `SettingsInversion` of the `Inversion` object that is created from the fit.
        use_preloaded_grid
            Certain pixelization's construct their mesh in the source-plane from a stochastic KMeans algorithm. This
            grid may be output to hard-disk after the model-fit and loaded via the database to ensure the same grid is
            used as the fit.
        """
        super().__init__(aggregator=aggregator)

        self.settings_dataset = settings_dataset
        self.settings_inversion = settings_inversion
        self.use_preloaded_grid = use_preloaded_grid

    def object_via_gen_from(self, fit, galaxies) -> List[FitImaging]:
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
            settings_inversion=self.settings_inversion,
            use_preloaded_grid=self.use_preloaded_grid,
        )
