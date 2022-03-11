import logging
import os
from typing import Callable, Union

from autoconf import conf

from autofit.exc import PriorLimitException

import autofit as af

from autolens.imaging.fit_imaging import FitImaging
from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.lens.model.preloads import Preloads

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class FitMaker:
    def __init__(self, model: af.Collection, fit_func: Callable):
        """
        Makes fits using an input PyAutoFit `model`, where the parameters of the model are drawn from its prior. This
        uses an input `fit_func`, which given an `instance` of the model creates the fit object.

        This is used for implicit preloading in the `Analysis` classes, whereby the created fits are compared against
        one another to determine whether certain components of the analysis can be preloaded.

        This includes functionality for creating the fit via the model in different ways, so that if certain
        models are ill-defined another is used instead.

        Parameters
        ----------
        model
            A **PyAutoFit** model object which via its parameters and their priors can created instances of the model.
        fit_func
            A function which given the instance of the model creates a `Fit` object.
        """

        self.model = model
        self.fit_func = fit_func

    def fit_via_model_from(
        self, unit_value: float
    ) -> Union[FitImaging, FitInterferometer]:
        """
        Create a fit via the model.

        This first tries to compute the fit from the input `unit_value`, where the `unit_value` defines unit hyper
        cube values of each parameter's prior in the model, used to map each value to physical values for the fit.

        If this model fit produces an `Exception` because the parameter combination does not fit the data accurately,
        a sequence of random fits are instead used into an exception is not returned. However, if the number
        of `preload_attempts` defined in the configuration files is exceeded a None is returned.

        Parameters
        ----------
        unit_value
            The unit hyper cube values of each parameter's prior in the model, used to map each value to physical
            values for the fit.

        Returns
        -------
        fit
            A fit object where an instance of the model has been fitted to the data.
        """
        try:
            try:
                return self.fit_unit_instance_from(unit_value=unit_value)
            except IndexError as e:
                raise Exception from e
        except (Exception, PriorLimitException):
            return self.fit_random_instance_from()

    def fit_unit_instance_from(
        self, unit_value: float
    ) -> Union[FitImaging, FitInterferometer]:
        """
        Create a fit via the model using an input `unit_value`, where the `unit_value` defines unit hyper
        cube values of each parameter's prior in the model, used to map each value to physical values for the fit.

        Parameters
        ----------
        unit_value
            The unit hyper cube values of each parameter's prior in the model, used to map each value to physical
            values for the fit.

        Returns
        -------
        fit
            A fit object where an instance of the model has been fitted to the data.
        """
        ignore_prior_limits = conf.instance["general"]["model"]["ignore_prior_limits"]
        conf.instance["general"]["model"]["ignore_prior_limits"] = True

        instance = self.model.instance_from_unit_vector(
            unit_vector=[unit_value] * self.model.prior_count
        )

        conf.instance["general"]["model"]["ignore_prior_limits"] = ignore_prior_limits

        return self.fit_func(
            instance=instance,
            preload_overwrite=Preloads(use_w_tilde=False),
            check_positions=False,
        )

    def fit_random_instance_from(self) -> Union[FitImaging, FitInterferometer]:
        """
        Create a fit via the model by guessing a  a sequence of random fits until an exception is not returned. If
        the number of `preload_attempts` defined in the configuration files is exceeded a None is returned.

        Returns
        -------
        fit
            A fit object where an instance of the model has been fitted to the data.
        """
        preload_attempts = conf.instance["general"]["analysis"]["preload_attempts"]

        ignore_prior_limits = conf.instance["general"]["model"]["ignore_prior_limits"]
        conf.instance["general"]["model"]["ignore_prior_limits"] = True

        for i in range(preload_attempts):

            try:

                instance = self.model.random_instance()

                conf.instance["general"]["model"][
                    "ignore_prior_limits"
                ] = ignore_prior_limits

                return self.fit_func(
                    instance=instance,
                    preload_overwrite=Preloads(use_w_tilde=False),
                    check_positions=False,
                )
            except Exception:
                pass

            if i == preload_attempts:

                conf.instance["general"]["model"][
                    "ignore_prior_limits"
                ] = ignore_prior_limits
