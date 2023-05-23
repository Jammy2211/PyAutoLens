from typing import List

import autofit as af
import autogalaxy as ag

from autogalaxy.aggregator.abstract import AbstractAgg
from autolens.lens.ray_tracing import Tracer


def _tracer_from(fit: af.Fit, galaxies: List[ag.Galaxy]) -> Tracer:
    """
    Returns a `Tracer` object from a PyAutoFit database `Fit` object and an instance of galaxies from a non-linear
    search model-fit.

    This function adds the `adapt_model_image` and `adapt_galaxy_image_path_dict` to the galaxies before constructing
    the `Tracer`, if they were used.

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of model-fits.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.

    Returns
    -------
    Tracer
        The tracer computed via an instance of galaxies.
    """

    adapt_model_image = fit.value(name="adapt_model_image")
    adapt_galaxy_image_path_dict = fit.value(name="adapt_galaxy_image_path_dict")

    galaxies_with_adapt = []

    if adapt_galaxy_image_path_dict is not None:
        galaxy_path_list = [
            gal[0] for gal in fit.instance.path_instance_tuples_for_class(ag.Galaxy)
        ]

        for galaxy_path, galaxy in zip(galaxy_path_list, galaxies):
            if galaxy_path in adapt_galaxy_image_path_dict:
                galaxy.adapt_model_image = adapt_model_image
                galaxy.adapt_galaxy_image = adapt_galaxy_image_path_dict[galaxy_path]

            galaxies_with_adapt.append(galaxy)

        return Tracer.from_galaxies(galaxies=galaxies_with_adapt)

    return Tracer.from_galaxies(galaxies=galaxies)


class TracerAgg(AbstractAgg):
    """
    Wraps a PyAutoFit aggregator in order to create generators of tracers corresponding to the results of a non-linear
    search model-fit.
    """

    def object_via_gen_from(self, fit, galaxies) -> Tracer:
        """
        Creates a `Tracer` object from a `ModelInstance` that contains the galaxies of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of model-fits.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.

        Returns
        -------
        Tracer
            A tracer whose galaxies are a sample of a PyAutoFit non-linear search.
        """
        return _tracer_from(fit=fit, galaxies=galaxies)
