import autofit as af
import autolens as al

from autofit import exc

from autofit.database.model.fit import Fit
from autogalaxy.analysis.aggregator.aggregator import (
    _imaging_from,
    _interferometer_from,
)

from functools import partial

from typing import Optional, List, Generator, Tuple


def _tracer_from(fit: Fit, galaxies: List[al.Galaxy]) -> "al.Tracer":
    """
    Returns a `Tracer` object from a PyAutoFit database `Fit` object and an instance of galaxies from a non-linear
    search model-fit.

    This function adds the `hyper_model_image` and `hyper_galaxy_image_path_dict` to the galaxies before constructing
    the `Tracer`, if they were used.

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of PyAutoGalaxy model-fits.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.

    Returns
    -------
    Tracer
        The tracer computed via an instance of galaxies.
    """

    hyper_model_image = fit.value(name="hyper_model_image")
    hyper_galaxy_image_path_dict = fit.value(name="hyper_galaxy_image_path_dict")

    galaxies_with_hyper = []

    if hyper_galaxy_image_path_dict is not None:

        galaxy_path_list = [
            gal[0] for gal in fit.instance.path_instance_tuples_for_class(al.Galaxy)
        ]

        for (galaxy_path, galaxy) in zip(galaxy_path_list, galaxies):

            if galaxy_path in hyper_galaxy_image_path_dict:
                galaxy.hyper_model_image = hyper_model_image
                galaxy.hyper_galaxy_image = hyper_galaxy_image_path_dict[galaxy_path]

            galaxies_with_hyper.append(galaxy)

        return al.Tracer.from_galaxies(galaxies=galaxies_with_hyper)

    return al.Tracer.from_galaxies(galaxies=galaxies)


def _fit_imaging_from(
    fit: Fit,
    galaxies: List[al.Galaxy],
    settings_imaging: al.SettingsImaging = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
    use_preloaded_grid: bool = True,
) -> "al.FitImaging":
    """
    Returns a `FitImaging` object from a PyAutoFit database `Fit` object and an instance of galaxies from a non-linear
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
    FitImaging
        The fit to the imaging dataset computed via an instance of galaxies.        
    """

    imaging = _imaging_from(fit=fit, settings_imaging=settings_imaging)

    tracer = _tracer_from(fit=fit, galaxies=galaxies)

    settings_pixelization = settings_pixelization or fit.value(
        name="settings_pixelization"
    )
    settings_inversion = settings_inversion or fit.value(name="settings_inversion")

    preloads = al.Preloads()

    if use_preloaded_grid:

        sparse_grids_of_planes = fit.value(name="preload_sparse_grids_of_planes")

        if sparse_grids_of_planes is not None:

            preloads = al.Preloads(sparse_grids_of_planes=sparse_grids_of_planes)

    return al.FitImaging(
        imaging=imaging,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
        preloads=preloads,
    )


def _fit_interferometer_from(
    fit: Fit,
    galaxies: List[al.Galaxy],
    real_space_mask: Optional[al.Mask2D] = None,
    settings_interferometer: al.SettingsInterferometer = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
    use_preloaded_grid: bool = True,
) -> "al.FitInterferometer":
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

            preloads = al.Preloads(sparse_grids_of_planes=sparse_grids_of_planes)

    return al.FitInterferometer(
        interferometer=interferometer,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
        preloads=preloads,
    )


class AbstractAgg:
    def __init__(self, aggregator: af.Aggregator):
        """
        An abstract aggregator wrapper, which makes it straight forward to compute generators of objects from specific
        samples of a non-linear search.

        For example, in **PyAutoLens**, this makes it straight forward to create generators of `Tracer`'s drawn from
        the PDF estimated by a non-linear for efficient error calculation of derived quantities.
        Parameters
        ----------
        aggregator
            An PyAutoFit aggregator containing the results of non-linear searches performed by PyAutoFit.
        """
        self.aggregator = aggregator

    def make_object_for_gen(self, fit: Fit, galaxies: List[al.Galaxy]) -> object:
        """
        Abstract method that is overwritten with functions that create the generator of an object from the results
        of a non-linear search.

        For example, in the `TracerAgg` object, this function is overwritten such that it creates a `Tracer` from a
        `ModelInstance` that contains the galaxies of a sample from a non-linear search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of PyAutoGalaxy model-fits.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.

        Returns
        -------
        Generator
            A generator that creates an object used in the model-fitting process of a non-linear search.
        """
        raise NotImplementedError

    def max_log_likelihood_gen(self) -> Generator:
        """
        Returns a generator using the maximum likelihood instance of a non-linear search.

        This generator creates a list containing the maximum log instance of every result loaded in the aggregator.

        For example, in **PyAutoLens**, by overwriting the `make_gen_from_object` method this returns a generator
        of `Tracer` objects from a PyAutoFit aggregator. This generator then generates a list of the maximum log
        likelihood `Tracer` objects for all aggregator results.
        """

        def func_gen(fit: Fit) -> Generator:
            return self.make_object_for_gen(fit=fit, galaxies=fit.instance.galaxies)

        return self.aggregator.map(func=func_gen)

    def all_above_weight_gen(self, minimum_weight: float) -> Generator:
        """
        Returns a generator which for every result generates a list of objects whose parameter values are all those
        in the non-linear search with a weight about an input `minimum_weight` value. This enables straight forward
        error estimation.

        This generator creates lists containing instances whose non-linear sample weight are above the value of
        `minimum_weight`. For example, if the aggregator contains 10 results and each result has 100 samples above the
        `minimum_weight`, a list of 10 entries will be returned, where each entry in this list contains 100 object's
        paired with each non-linear sample.

        For example, in **PyAutoLens**, by overwriting the `make_gen_from_object` method this returns a generator
        of `Tracer` objects from a PyAutoFit aggregator. This generator then generates lists of `Tracer` objects
        corresponding to all non-linear search samples above the `minimum_weight`.

        Parameters
        ----------
        minimum_weight
            The minimum weight of a non-linear sample, such that samples with a weight below this value are discarded
            and not included in the generator.
        """

        def func_gen(fit: Fit, minimum_weight: float) -> List[object]:

            samples = fit.value(name="samples")

            all_above_weight_list = []

            for sample in samples.sample_list:

                if sample.weight > minimum_weight:
                    instance = sample.instance_for_model(model=samples.model)

                    all_above_weight_list.append(
                        self.make_object_for_gen(fit=fit, galaxies=instance.galaxies)
                    )

            return all_above_weight_list

        func = partial(func_gen, minimum_weight=minimum_weight)

        return self.aggregator.map(func=func)

    def randomly_drawn_from_pdf_gen(self, total_samples: int):
        """
        Returns a generator which for every result generates a list of objects whose parameter values are drawn
        randomly from the PDF. This enables straight forward error estimation.

        This generator creates lists containing instances that are drawn randomly from the PDF for every result loaded
        in the aggregator. For example, the aggregator contains 10 results and if `total_samples=100`, a list of 10
        entries will be returned, where each entry in this list contains 100 object's paired with non-linear samples
        randomly drawn from the PDF.

        For example, in **PyAutoLens**, by overwriting the `make_gen_from_object` method this returns a generator
        of `Tracer` objects from a PyAutoFit aggregator. This generator then generates lists of `Tracer` objects
        corresponding to non-linear search samples randomly drawn from the PDF.

        Parameters
        ----------
        total_samples
            The total number of non-linear search samples that should be randomly drawn from the PDF.
        """

        def func_gen(fit: Fit, total_samples: int) -> List[object]:

            samples = fit.value(name="samples")

            return [
                self.make_object_for_gen(
                    fit=fit,
                    galaxies=samples.instance_drawn_randomly_from_pdf().galaxies,
                )
                for i in range(total_samples)
            ]

        func = partial(func_gen, total_samples=total_samples)

        return self.aggregator.map(func=func)


class TracerAgg(AbstractAgg):
    """
    Wraps a PyAutoFit aggregator in order to create generators of tracers corresponding to the results of a non-linear
    search model-fit.
    """

    def make_object_for_gen(self, fit, galaxies) -> "al.Tracer":
        """
        Creates a `Tracer` object from a `ModelInstance` that contains the galaxies of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of PyAutoGalaxy model-fits.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.

        Returns
        -------
        Tracer
            A tracer whose galaxies are a sample of a PyAutoFit non-linear search.
        """
        return _tracer_from(fit=fit, galaxies=galaxies)


class FitImagingAgg(AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        settings_imaging: Optional[al.SettingsImaging] = None,
        settings_pixelization: Optional[al.SettingsPixelization] = None,
        settings_inversion: Optional[al.SettingsInversion] = None,
        use_preloaded_grid: bool = True,
    ):
        """
        Wraps a PyAutoFit aggregator in order to create generators of fits to imaging data, corresponding to the
        results of a non-linear search model-fit.
        """
        super().__init__(aggregator=aggregator)

        self.settings_imaging = settings_imaging
        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion
        self.use_preloaded_grid = use_preloaded_grid

    def make_object_for_gen(self, fit, galaxies) -> "al.FitImaging":
        """
        Creates a `FitImaging` object from a `ModelInstance` that contains the galaxies of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of PyAutoGalaxy model-fits.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.

        Returns
        -------
        FitImaging
            A fit to imaging data whose galaxies are a sample of a PyAutoFit non-linear search.
        """
        return _fit_imaging_from(
            fit=fit,
            galaxies=galaxies,
            settings_imaging=self.settings_imaging,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            use_preloaded_grid=self.use_preloaded_grid,
        )


class FitInterferometerAgg(AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        settings_interferometer: Optional[al.SettingsInterferometer] = None,
        settings_pixelization: Optional[al.SettingsPixelization] = None,
        settings_inversion: Optional[al.SettingsInversion] = None,
        use_preloaded_grid: bool = True,
        real_space_mask: Optional[al.Mask2D] = None,
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

    def make_object_for_gen(self, fit, galaxies) -> "al.FitInterferometer":
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


class SubhaloAgg:
    def __init__(
        self,
        #  aggregator_no_subhalo: af.Aggregator,
        aggregator_grid_search: af.GridSearchAggregator,
        settings_imaging: Optional[al.SettingsImaging] = None,
        settings_pixelization: Optional[al.SettingsPixelization] = None,
        settings_inversion: Optional[al.SettingsInversion] = None,
        use_preloaded_grid: bool = True,
    ):
        """
        Wraps a PyAutoFit aggregator in order to create generators of fits to imaging data, corresponding to the
        results of a non-linear search model-fit.
        """
        #        self.aggregator_no_subhalo = aggregator_no_subhalo
        self.aggregator_grid_search = aggregator_grid_search
        self.settings_imaging = settings_imaging
        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion
        self.use_preloaded_grid = use_preloaded_grid

        if len(aggregator_grid_search) == 0:
            raise exc.AggregatorException(
                "There is no grid search of results in the aggregator."
            )
        elif len(aggregator_grid_search) > 1:
            raise exc.AggregatorException(
                "There is more than one grid search of results in the aggregator - please filter the"
                "aggregator."
            )

    @property
    def grid_search_result(self) -> af.GridSearchResult:
        return self.aggregator_grid_search[0]["result"]
