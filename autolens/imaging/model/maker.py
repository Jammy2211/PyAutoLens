import logging
import os

from autoconf import conf

from autolens.lens.model.preloads import Preloads

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class FitImagingMaker:
    def __init__(self, model, fit_from_instance_func):

        self.model = model
        self.fit_from_instance_func = fit_from_instance_func

    def fit_via_model(self, unit_value):

        try:
            return self.fit_from_unit_instance(unit_value=unit_value)
        except Exception:
            return self.fit_from_random_instance()

    def fit_from_unit_instance(self, unit_value):

        ignore_prior_limits = conf.instance["general"]["model"]["ignore_prior_limits"]
        conf.instance["general"]["model"]["ignore_prior_limits"] = True

        instance = self.model.instance_from_unit_vector(
            unit_vector=[unit_value] * self.model.prior_count
        )

        conf.instance["general"]["model"]["ignore_prior_limits"] = ignore_prior_limits

        return self.fit_from_instance_func(
            instance=instance,
            preload_overwrite=Preloads(use_w_tilde=False),
            check_positions=False,
        )

    def fit_from_random_instance(self):

        preload_attempts = conf.instance["general"]["analysis"]["preload_attempts"]

        ignore_prior_limits = conf.instance["general"]["model"]["ignore_prior_limits"]
        conf.instance["general"]["model"]["ignore_prior_limits"] = True

        for i in range(preload_attempts):

            instance = self.model.random_instance()

            try:

                conf.instance["general"]["model"][
                    "ignore_prior_limits"
                ] = ignore_prior_limits

                return self.fit_from_instance_func(
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

                return

    def fit_via_prior_medians(self, preload_overwrite=None):

        ignore_prior_limits = conf.instance["general"]["model"]["ignore_prior_limits"]
        conf.instance["general"]["model"]["ignore_prior_limits"] = True

        instance = self.model.instance_from_prior_medians()

        conf.instance["general"]["model"]["ignore_prior_limits"] = ignore_prior_limits

        return self.fit_from_instance_func(
            instance=instance,
            check_positions=False,
            preload_overwrite=preload_overwrite,
        )
