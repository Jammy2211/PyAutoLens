import autolens as al

import numpy as np
import os


def tracer_generator_from_aggregator(aggregator):
    return aggregator.map(func=tracer_from_agg_obj)


def tracer_from_agg_obj(agg_obj):

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


def masked_imaging_generator_from_aggregator(aggregator):
    return aggregator.map(func=masked_imaging_from_agg_obj)


def masked_imaging_from_agg_obj(agg_obj):

    return al.MaskedImaging(
        imaging=agg_obj.dataset,
        mask=agg_obj.mask,
        psf_shape_2d=agg_obj.meta_dataset.psf_shape_2d,
        pixel_scale_interpolation_grid=agg_obj.meta_dataset.pixel_scale_interpolation_grid,
        inversion_pixel_limit=agg_obj.meta_dataset.inversion_pixel_limit,
        inversion_uses_border=agg_obj.meta_dataset.inversion_uses_border,
        positions_threshold=agg_obj.meta_dataset.positions_threshold,
    )


def fit_imaging_generator_from_aggregator(aggregator):
    return aggregator.map(func=fit_imaging_from_agg_obj)


def fit_imaging_from_agg_obj(agg_obj):

    masked_imaging = masked_imaging_from_agg_obj(agg_obj=agg_obj)
    tracer = tracer_from_agg_obj(agg_obj=agg_obj)

    return al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)


def results_array_from_grid_phase_results(file_results):

    try:
        y, x, evidence = np.loadtxt(
            file_results, delimiter=",", skiprows=1, unpack=True
        )
    except:
        raise FileNotFoundError

    pixel_scale = np.abs(x[0] - x[1])

    shape_2d = (int(np.sqrt(evidence.shape)), int(np.sqrt(evidence.shape)))

    return al.Array.manual_1d(
        array=evidence, pixel_scales=(pixel_scale, pixel_scale), shape_2d=shape_2d
    )
