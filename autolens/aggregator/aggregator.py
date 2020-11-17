import autofit as af
import autolens as al

from autofit import exc
from functools import partial
import numpy as np


def tracer_generator_from_aggregator(aggregator: af.Aggregator):
    """
    Returns a generator of `Tracer` objects from an input aggregator, which generates a list of the `Tracer` objects
    for every set of results loaded in the aggregator.

    This is performed by mapping the *tracer_from_agg_obj* with the aggregator, which sets up each tracer using only
    generators ensuring that manipulating the planes of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""
    return aggregator.map(func=tracer_from_agg_obj)


def tracer_from_agg_obj(agg_obj: af.PhaseOutput) -> "al.Tracer":
    """
    Returns a `Tracer` object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe that
     it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator outputs
     such that the function can use the *Aggregator*'s map function to to create a `Tracer` generator.

     The `Tracer` is created following the same method as the PyAutoLens `Phase` classes using an instance of the
     maximum log likelihood model's galaxies. These galaxies have their hyper-images added (if they were used in the
     fit) and passed into a Tracer object.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoLens model-fits.
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


def masked_imaging_generator_from_aggregator(
    aggregator: af.Aggregator, settings_masked_imaging: al.SettingsMaskedImaging = None
):
    """
    Returns a generator of *MaskedImaging* objects from an input aggregator, which generates a list of the
    *MaskedImaging* objects for every set of results loaded in the aggregator.

    This is performed by mapping the *masked_imaging_from_agg_obj* with the aggregator, which sets up each masked
    imaging using only generators ensuring that manipulating the masked imaging of large sets of results is done in a
    memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""
    func = partial(
        masked_imaging_from_agg_obj, settings_masked_imaging=settings_masked_imaging
    )
    return aggregator.map(func=func)


def masked_imaging_from_agg_obj(
    agg_obj: af.PhaseOutput, settings_masked_imaging: al.SettingsMaskedImaging = None
) -> "al.MaskedImaging":
    """
    Returns a *MaskedImaging* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe
     that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator
     outputs such that the function can use the *Aggregator*'s map function to to create a *MaskedImaging* generator.

     The *MaskedImaging* is created following the same method as the PyAutoLens `Phase` classes, including using the
     *meta_dataset* instance output by the phase to load inputs of the *MaskedImaging* (e.g. psf_shape_2d).

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoLens model-fits.
    """

    if settings_masked_imaging is None:
        settings_masked_imaging = agg_obj.settings.settings_masked_imaging

    return al.MaskedImaging(
        imaging=agg_obj.dataset, mask=agg_obj.mask, settings=settings_masked_imaging
    )


def fit_imaging_generator_from_aggregator(
    aggregator: af.Aggregator,
    settings_masked_imaging: al.SettingsMaskedImaging = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
):
    """
    Returns a generator of `FitImaging` objects from an input aggregator, which generates a list of the
    `FitImaging` objects for every set of results loaded in the aggregator.

    This is performed by mapping the *fit_imaging_from_agg_obj* with the aggregator, which sets up each fit using
    only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""

    func = partial(
        fit_imaging_from_agg_obj,
        settings_masked_imaging=settings_masked_imaging,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )

    return aggregator.map(func=func)


def fit_imaging_from_agg_obj(
    agg_obj: af.PhaseOutput,
    settings_masked_imaging: al.SettingsMaskedImaging = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
) -> "al.FitImaging":
    """
    Returns a `FitImaging` object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe
     that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator
     outputs such that the function can use the *Aggregator*'s map function to to create a `FitImaging` generator.

     The `FitImaging` is created following the same method as the PyAutoLens `Phase` classes.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoLens model-fits.
    """
    masked_imaging = masked_imaging_from_agg_obj(
        agg_obj=agg_obj, settings_masked_imaging=settings_masked_imaging
    )
    tracer = tracer_from_agg_obj(agg_obj=agg_obj)

    if settings_pixelization is None:
        settings_pixelization = agg_obj.settings.settings_pixelization

    if settings_inversion is None:
        settings_inversion = agg_obj.settings.settings_inversion

    return al.FitImaging(
        masked_imaging=masked_imaging,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )


def masked_interferometer_generator_from_aggregator(
    aggregator: af.Aggregator,
    settings_masked_interferometer: al.SettingsMaskedInterferometer = None,
):
    """
    Returns a generator of *MaskedInterferometer* objects from an input aggregator, which generates a list of the
    *MaskedInterferometer* objects for every set of results loaded in the aggregator.

    This is performed by mapping the *masked_interferometer_from_agg_obj* with the aggregator, which sets up each masked
    interferometer object using only generators ensuring that manipulating the masked interferometer objects of large
    sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""
    func = partial(
        masked_interferometer_from_agg_obj,
        settings_masked_interferometer=settings_masked_interferometer,
    )
    return aggregator.map(func=func)


def masked_interferometer_from_agg_obj(
    agg_obj: af.PhaseOutput,
    settings_masked_interferometer: al.SettingsMaskedInterferometer = None,
) -> "al.MaskedInterferometer":
    """
    Returns a *MaskedInterferometer* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to
    describe that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's
    generator outputs such that the function can use the *Aggregator*'s map function to to create a
    *MaskedInterferometer* generator.

    The *MaskedInterferometer* is created following the same method as the PyAutoLens `Phase` classes, including
    using the *meta_dataset* instance output by the phase to load inputs of the *MaskedInterferometer*
    (e.g. psf_shape_2d).

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoLens model-fits.
    """

    if settings_masked_interferometer is None:
        settings_masked_interferometer = agg_obj.settings.settings_masked_interferometer

    return al.MaskedInterferometer(
        interferometer=agg_obj.dataset,
        visibilities_mask=agg_obj.mask,
        real_space_mask=agg_obj.attributes.real_space_mask,
        settings=settings_masked_interferometer,
    )


def fit_interferometer_generator_from_aggregator(
    aggregator: af.Aggregator,
    settings_masked_interferometer: al.SettingsMaskedInterferometer = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
):
    """
    Returns a *FitInterferometer* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to
    describe that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's
    generator outputs such that the function can use the *Aggregator*'s map function to to create a *FitInterferometer*
    generator.

    The *FitInterferometer* is created following the same method as the PyAutoLens `Phase` classes.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoLens model-fits.
    """

    func = partial(
        fit_interferometer_from_agg_obj,
        settings_masked_interferometer=settings_masked_interferometer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )
    return aggregator.map(func=func)


def fit_interferometer_from_agg_obj(
    agg_obj: af.PhaseOutput,
    settings_masked_interferometer: al.SettingsMaskedInterferometer = None,
    settings_pixelization: al.SettingsPixelization = None,
    settings_inversion: al.SettingsInversion = None,
) -> "al.FitInterferometer":
    """
    Returns a generator of *FitInterferometer* objects from an input aggregator, which generates a list of the
    *FitInterferometer* objects for every set of results loaded in the aggregator.

    This is performed by mapping the *fit_interferometer_from_agg_obj* with the aggregator, which sets up each fit
    using only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient
    way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""
    masked_interferometer = masked_interferometer_from_agg_obj(
        agg_obj=agg_obj, settings_masked_interferometer=settings_masked_interferometer
    )
    tracer = tracer_from_agg_obj(agg_obj=agg_obj)

    if settings_pixelization is None:
        settings_pixelization = agg_obj.settings.settings_pixelization

    if settings_inversion is None:
        settings_inversion = agg_obj.settings.settings_inversion

    return al.FitInterferometer(
        masked_interferometer=masked_interferometer,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )


def grid_search_result_as_array(
    aggregator: af.Aggregator, use_log_evidences: bool = True
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
        grid_search_result=grid_search_results[0], use_log_evidences=use_log_evidences
    )


def grid_search_subhalo_masses_as_array(aggregator: af.Aggregator) -> al.Array:

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


def grid_search_subhalo_centres_as_array(aggregator: af.Aggregator) -> al.Array:

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
    grid_search_result, use_log_evidences=True
) -> al.Array:

    if grid_search_result.no_dimensions != 2:
        raise exc.AggregatorException(
            "The GridSearchResult is not dimensions 2, meaning a 2D array cannot be made."
        )

    if use_log_evidences:
        values = [
            value
            for values in grid_search_result.log_evidence_values
            for value in values
        ]
    else:
        values = [
            value
            for values in grid_search_result.max_log_likelihood_values
            for value in values
        ]

    return al.Array.manual_yx_and_values(
        y=[centre[0] for centre in grid_search_result.physical_centres_lists],
        x=[centre[1] for centre in grid_search_result.physical_centres_lists],
        values=values,
        pixel_scales=grid_search_result.physical_step_sizes,
        shape_2d=grid_search_result.shape,
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

    return al.Array.manual_yx_and_values(
        y=[centre[0] for centre in grid_search_result.physical_centres_lists],
        x=[centre[1] for centre in grid_search_result.physical_centres_lists],
        values=masses,
        pixel_scales=grid_search_result.physical_step_sizes,
        shape_2d=grid_search_result.shape,
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
