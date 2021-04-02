import autofit as af
import autolens as al

from autofit import exc
from functools import partial
import numpy as np
from os import path
import json


def tracer_generator_from_aggregator(aggregator: af.Aggregator):
    """
    Returns a generator of `Tracer` objects from an input aggregator, which generates a list of the `Tracer` objects
    for every set of results loaded in the aggregator.

    This is performed by mapping the `tracer_from_agg_obj` with the aggregator, which sets up each tracer using only
    generators ensuring that manipulating the planes of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""
    return aggregator.map(func=tracer_from_agg_obj)


def tracer_from_agg_obj(agg_obj: af.SearchOutput) -> "al.Tracer":
    """
    Returns a `Tracer` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe that
     it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator outputs
     such that the function can use the `Aggregator`'s map function to to create a `Tracer` generator.

     The `Tracer` is created using an instance of the maximum log likelihood model's galaxies. These galaxies have 
     their hyper-images added (if they were used in the fit) and passed into a Tracer object.

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoLens model-fits.
    """
    samples = agg_obj.samples
    attributes = agg_obj.attributes
    max_log_likelihood_instance = samples.max_log_likelihood_instance
    galaxies = max_log_likelihood_instance.galaxies

    if attributes.hyper_galaxy_image_path_dict is not None:

        for (
            galaxy_path,
            galaxy,
        ) in max_log_likelihood_instance.path_instance_tuples_for_class(al.Galaxy):
            if galaxy_path in attributes.hyper_galaxy_image_path_dict:
                galaxy.hyper_model_image = attributes.hyper_model_image
                galaxy.hyper_galaxy_image = attributes.hyper_galaxy_image_path_dict[
                    galaxy_path
                ]

    return al.Tracer.from_galaxies(galaxies=galaxies)


def imaging_generator_from_aggregator(
    aggregator: af.Aggregator, settings_imaging: al.SettingsImaging = None
):
    """
    Returns a generator of `Imaging` objects from an input aggregator, which generates a list of the
    `Imaging` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `imaging_from_agg_obj` with the aggregator, which sets up each
    imaging using only generators ensuring that manipulating the imaging of large sets of results is done in a
    memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""
    func = partial(imaging_from_agg_obj, settings_imaging=settings_imaging)
    return aggregator.map(func=func)


def imaging_from_agg_obj(
    agg_obj: af.SearchOutput, settings_imaging: al.SettingsImaging = None
) -> "al.Imaging":
    """
    Returns a `Imaging` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
    that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
    outputs such that the function can use the `Aggregator`'s map function to to create a `Imaging` generator.

    The `Imaging` is created, including using the
    `meta_dataset` instance output by the Search to load inputs of the `Imaging` (e.g. psf_shape_2d).

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoLens model-fits.
    """

    imaging = agg_obj.dataset

    if settings_imaging is None:
        return imaging

    return imaging.apply_settings(settings=settings_imaging)


def fit_imaging_generator_from_aggregator(
    aggregator: af.Aggregator,
    settings_imaging: al.SettingsImaging = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
):
    """
    Returns a generator of `FitImaging` objects from an input aggregator, which generates a list of the
    `FitImaging` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `fit_imaging_from_agg_obj` with the aggregator, which sets up each fit using
    only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""

    func = partial(
        fit_imaging_from_agg_obj,
        settings_imaging=settings_imaging,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )

    return aggregator.map(func=func)


def fit_imaging_from_agg_obj(
    agg_obj: af.SearchOutput,
    settings_imaging: al.SettingsImaging = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
) -> "al.FitImaging":
    """
    Returns a `FitImaging` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
     that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
     outputs such that the function can use the `Aggregator`'s map function to to create a `FitImaging` generator.

     The `FitImaging` is created.

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoLens model-fits.
    """
    imaging = imaging_from_agg_obj(agg_obj=agg_obj, settings_imaging=settings_imaging)
    tracer = tracer_from_agg_obj(agg_obj=agg_obj)

    if settings_pixelization is None:
        settings_pixelization = agg_obj.settings_pixelization

    if settings_inversion is None:
        settings_inversion = agg_obj.settings_inversion

    return al.FitImaging(
        imaging=imaging,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )


def interferometer_generator_from_aggregator(
    aggregator: af.Aggregator, settings_interferometer: al.SettingsInterferometer = None
):
    """
    Returns a generator of `Interferometer` objects from an input aggregator, which generates a list of the
    `Interferometer` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `interferometer_from_agg_obj` with the aggregator, which sets up each
    interferometer object using only generators ensuring that manipulating the interferometer objects of large
    sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""
    func = partial(
        interferometer_from_agg_obj, settings_interferometer=settings_interferometer
    )
    return aggregator.map(func=func)


def interferometer_from_agg_obj(
    agg_obj: af.SearchOutput, settings_interferometer: al.SettingsInterferometer = None
) -> "al.Interferometer":
    """
    Returns a `Interferometer` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to
    describe that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's
    generator outputs such that the function can use the `Aggregator`'s map function to to create a
    `Interferometer` generator.

    The `Interferometer` is created, including
    using the `meta_dataset` instance output by the Search to load inputs of the `Interferometer`
    (e.g. psf_shape_2d).

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoLens model-fits.
    """

    interferometer = agg_obj.dataset

    if settings_interferometer is None:
        return interferometer

    return interferometer.apply_settings(settings=settings_interferometer)


def fit_interferometer_generator_from_aggregator(
    aggregator: af.Aggregator,
    settings_interferometer: al.SettingsInterferometer = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
):
    """
    Returns a `FitInterferometer` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to
    describe that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's
    generator outputs such that the function can use the `Aggregator`'s map function to to create a `FitInterferometer`
    generator.

    The `FitInterferometer` is created.

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoLens model-fits.
    """

    func = partial(
        fit_interferometer_from_agg_obj,
        settings_interferometer=settings_interferometer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )
    return aggregator.map(func=func)


def fit_interferometer_from_agg_obj(
    agg_obj: af.SearchOutput,
    settings_interferometer: al.SettingsInterferometer = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
) -> "al.FitInterferometer":
    """
    Returns a generator of `FitInterferometer` objects from an input aggregator, which generates a list of the
    `FitInterferometer` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `fit_interferometer_from_agg_obj` with the aggregator, which sets up each fit
    using only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient
    way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits.
    """
    interferometer = interferometer_from_agg_obj(
        agg_obj=agg_obj, settings_interferometer=settings_interferometer
    )
    tracer = tracer_from_agg_obj(agg_obj=agg_obj)

    if settings_pixelization is None:
        settings_pixelization = agg_obj.settings_pixelization

    if settings_inversion is None:
        settings_inversion = agg_obj.settings_inversion

    return al.FitInterferometer(
        interferometer=interferometer,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )


def grid_search_result_as_array(
    aggregator: af.Aggregator,
    use_log_evidences: bool = True,
    use_stochastic_log_evidences: bool = False,
) -> np.ndarray:

    grid_search_result_gen = aggregator.values("grid_search_result")

    grid_search_results = list(filter(None, list(grid_search_result_gen)))

    if len(grid_search_results) == 0:
        raise exc.AggregatorException(
            "There is no grid search resultin the aggregator."
        )
    elif len(grid_search_results) > 1:
        raise exc.AggregatorException(
            "There is more than one grid search result in the aggregator - please filter the"
            "aggregator."
        )

    return grid_search_log_evidences_as_array_from_grid_search_result(
        grid_search_result=grid_search_results[0],
        use_log_evidences=use_log_evidences,
        use_stochastic_log_evidences=use_stochastic_log_evidences,
    )


def grid_search_subhalo_masses_as_array(aggregator: af.Aggregator) -> al.Array2D:

    grid_search_result_gen = aggregator.values("grid_search_result")

    grid_search_results = list(filter(None, list(grid_search_result_gen)))

    if len(grid_search_results) != 1:
        raise exc.AggregatorException(
            "There is more than one grid search result in the aggregator - please filter the"
            "aggregator."
        )

    return grid_search_subhalo_masses_as_array_from_grid_search_result(
        grid_search_result=grid_search_results[0]
    )


def grid_search_subhalo_centres_as_array(aggregator: af.Aggregator) -> al.Array2D:

    grid_search_result_gen = aggregator.values("grid_search_result")

    grid_search_results = list(filter(None, list(grid_search_result_gen)))

    if len(grid_search_results) != 1:
        raise exc.AggregatorException(
            "There is more than one grid search result in the aggregator - please filter the"
            "aggregator."
        )

    return grid_search_subhalo_masses_as_array_from_grid_search_result(
        grid_search_result=grid_search_results[0]
    )


def grid_search_log_evidences_as_array_from_grid_search_result(
    grid_search_result,
    use_log_evidences=True,
    use_stochastic_log_evidences: bool = False,
) -> al.Array2D:

    if grid_search_result.no_dimensions != 2:
        raise exc.AggregatorException(
            "The GridSearchResult is not dimensions 2, meaning a 2D array cannot be made."
        )

    if use_log_evidences and not use_stochastic_log_evidences:
        values = [
            value
            for values in grid_search_result.log_evidence_values
            for value in values
        ]
    elif use_stochastic_log_evidences:

        stochastic_log_evidences = []

        for result in grid_search_result.results:

            stochastic_log_evidences_json_file = path.join(
                result.search.paths.output_path, "stochastic_log_evidences.json"
            )

            try:
                with open(stochastic_log_evidences_json_file, "r") as f:
                    stochastic_log_evidences_array = np.asarray(json.load(f))
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"File not found at {result.search.paths.output_path}"
                )

            stochastic_log_evidences.append(np.median(stochastic_log_evidences_array))

        values = stochastic_log_evidences

    else:
        values = [
            value
            for values in grid_search_result.max_log_likelihood_values
            for value in values
        ]

    return al.Array2D.manual_yx_and_values(
        y=[centre[0] for centre in grid_search_result.physical_centres_lists],
        x=[centre[1] for centre in grid_search_result.physical_centres_lists],
        values=values,
        pixel_scales=grid_search_result.physical_step_sizes,
        shape_native=grid_search_result.shape,
    )


def grid_search_subhalo_masses_as_array_from_grid_search_result(
    grid_search_result,
) -> [float]:

    if grid_search_result.no_dimensions != 2:
        raise exc.AggregatorException(
            "The GridSearchResult is not dimensions 2, meaning a 2D array cannot be made."
        )

    masses = [
        res.samples.median_pdf_instance.galaxies.subhalo.mass.mass_at_200
        for results in grid_search_result.results_reshaped
        for res in results
    ]

    return al.Array2D.manual_yx_and_values(
        y=[centre[0] for centre in grid_search_result.physical_centres_lists],
        x=[centre[1] for centre in grid_search_result.physical_centres_lists],
        values=masses,
        pixel_scales=grid_search_result.physical_step_sizes,
        shape_native=grid_search_result.shape,
    )


def grid_search_subhalo_centres_as_array_from_grid_search_result(
    grid_search_result,
) -> [(float, float)]:

    if grid_search_result.no_dimensions != 2:
        raise exc.AggregatorException(
            "The GridSearchResult is not dimensions 2, meaning a 2D array cannot be made."
        )

    return [
        res.samples.median_pdf_instance.galaxies.subhalo.mass.centre
        for results in grid_search_result.results_reshaped
        for res in results
    ]
