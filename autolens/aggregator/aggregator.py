import autolens as al

from functools import partial


def tracer_generator_from_aggregator(aggregator):
    """Compute a generator of *Tracer* objects from an input aggregator, which generates a list of the *Tracer* objects 
    for every set of results loaded in the aggregator.

    This is performed by mapping the *tracer_from_agg_obj* with the aggregator, which sets up each tracer using only
    generators ensuring that manipulating the planes of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoLens model-fits."""
    return aggregator.map(func=tracer_from_agg_obj)


def tracer_from_agg_obj(agg_obj):
    """Compute a *Tracer* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe that
     it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator outputs
     such that the function can use the *Aggregator*'s map function to to create a *Tracer* generator.

     The *tracer* is created following the same method as the PyAutoLens *Phase* classes using an instance of the
     maximum log likelihood model's galaxies. These galaxies have their hyper-images added (if they were used in the
     fit) and passed into a Tracer object.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoLens model-fits.
    """
    samples = agg_obj.samples
    phase_attributes = agg_obj.phase_attributes
    max_log_likelihood_instance = samples.max_log_likelihood_instance
    galaxies = max_log_likelihood_instance.galaxies

    if phase_attributes.hyper_galaxy_image_path_dict is not None:

        for (
            galaxy_path,
            galaxy,
        ) in max_log_likelihood_instance.path_instance_tuples_for_class(al.Galaxy):
            if galaxy_path in phase_attributes.hyper_galaxy_image_path_dict:
                galaxy.hyper_model_image = phase_attributes.hyper_model_image
                galaxy.hyper_galaxy_image = phase_attributes.hyper_galaxy_image_path_dict[
                    galaxy_path
                ]

    return al.Tracer.from_galaxies(galaxies=galaxies)


def masked_imaging_generator_from_aggregator(aggregator, settings_masked_imaging=None):
    """Compute a generator of *MaskedImaging* objects from an input aggregator, which generates a list of the 
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


def masked_imaging_from_agg_obj(agg_obj, settings_masked_imaging=None):
    """Compute a *MaskedImaging* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe 
     that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator 
     outputs such that the function can use the *Aggregator*'s map function to to create a *MaskedImaging* generator.

     The *MaskedImaging* is created following the same method as the PyAutoLens *Phase* classes, including using the
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
    aggregator,
    settings_masked_imaging=None,
    settings_pixelization=None,
    settings_inversion=None,
):
    """Compute a generator of *FitImaging* objects from an input aggregator, which generates a list of the 
    *FitImaging* objects for every set of results loaded in the aggregator.

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
    agg_obj,
    settings_masked_imaging=None,
    settings_pixelization=None,
    settings_inversion=None,
):
    """Compute a *FitImaging* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe 
     that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator 
     outputs such that the function can use the *Aggregator*'s map function to to create a *FitImaging* generator.

     The *FitImaging* is created following the same method as the PyAutoLens *Phase* classes. 

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
    aggregator, settings_masked_interferometer=None
):
    """Compute a generator of *MaskedInterferometer* objects from an input aggregator, which generates a list of the 
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


def masked_interferometer_from_agg_obj(agg_obj, settings_masked_interferometer=None):
    """Compute a *MaskedInterferometer* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to 
    describe that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's 
    generator outputs such that the function can use the *Aggregator*'s map function to to create a 
    *MaskedInterferometer* generator.

    The *MaskedInterferometer* is created following the same method as the PyAutoLens *Phase* classes, including 
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
        real_space_mask=agg_obj.phase_attributes.real_space_mask,
        settings=settings_masked_interferometer,
    )


def fit_interferometer_generator_from_aggregator(
    aggregator,
    settings_masked_interferometer=None,
    settings_pixelization=None,
    settings_inversion=None,
):
    """Compute a *FitInterferometer* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to 
    describe that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's 
    generator outputs such that the function can use the *Aggregator*'s map function to to create a *FitInterferometer* 
    generator.

    The *FitInterferometer* is created following the same method as the PyAutoLens *Phase* classes. 

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
    agg_obj,
    settings_masked_interferometer=None,
    settings_pixelization=None,
    settings_inversion=None,
):
    """Compute a generator of *FitInterferometer* objects from an input aggregator, which generates a list of the 
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
