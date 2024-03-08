import logging
from typing import List, Optional

import autofit as af

from autolens.lens.tracer import Tracer

logger = logging.getLogger(__name__)


def _tracer_from(
    fit: af.Fit, instance: Optional[af.ModelInstance] = None
) -> List[Tracer]:
    """
    Returns a list of `Tracer` objects from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).
    - The adapt images associated with adaptive galaxy features (`adapt` folder).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Tracer` object for a given non-linear search sample
    (e.g. the maximum likelihood model). This includes associating adapt images with their respective galaxies.

    If multiple `Tracer` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of Tracers. This is necessary if each Tracer has different galaxies (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    instance
        A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
        randomly from the PDF).
    """

    if instance is None:
        instance = fit.instance

    try:
        cosmology = instance.cosmology
    except AttributeError:
        cosmology = fit.value(name="cosmology")

    tracer = Tracer(galaxies=instance.galaxies, cosmology=cosmology)

    if len(fit.children) > 0:
        logger.info(
            """
            Using database for a fit with multiple summed Analysis objects.

            Tracer objects do not fully support this yet (e.g. model parameters which vary over analyses may be incorrect)
            so proceed with caution!
            """
        )

        return [tracer] * len(fit.children)

    return [tracer]


class TracerAgg(af.AggBase):
    """
    Interfaces with an `PyAutoFit` aggregator object to create instances of `Tracer` objects from the results
    of a model-fit.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).
    - The adapt images associated with adaptive galaxy features (`adapt` folder).

    The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
    can load them all at once and create an `Tracer` object via the `_tracer_from` method.

    This class's methods returns generators which create the instances of the `Tracer` objects. This ensures
    that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
    `Tracer` instances in the memory at once.

    For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
    creates instances of the corresponding 3 `Tracer` objects.

    If multiple `Tracer` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of Tracers. This is necessary if each Tracer has different galaxies (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    This can be done manually, but this object provides a more concise API.

    Parameters
    ----------
    aggregator
        A `PyAutoFit` aggregator object which can load the results of model-fits.
    """

    def object_via_gen_from(
        self, fit, instance: Optional[af.ModelInstance] = None
    ) -> List[Tracer]:
        """
        Returns a generator of `Tracer` objects from an input aggregator.

        See `__init__` for a description of how the `Tracer` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.
        """
        return _tracer_from(fit=fit, instance=instance)
