from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from autogalaxy import mock


import autofit as af


class MockResult(af.m.MockResult):
    def __init__(
        self,
        samples: mock.MockSamples = None,
        instance: af.Instance = None,
        model: af.Model = None,
        analysis: mock.MockAnalysis = None,
        search: af.mock.MockSearch = None,
        mask=None,
        model_image=None,
        max_log_likelihood_tracer=None,
        max_log_likelihood_fit=None,
        max_log_likelihood_mesh_grids_of_planes=None,
        adapt_galaxy_image_path_dict=None,
        adapt_model_image=None,
        pixelization=None,
        positions=None,
        updated_positions=None,
        updated_positions_threshold=None,
        stochastic_log_likelihoods=None,
    ):
        super().__init__(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
        )

        self.positions = None
        self.mask = mask
        self.adapt_galaxy_image_path_dict = adapt_galaxy_image_path_dict
        self.adapt_model_image = adapt_model_image
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.max_log_likelihood_tracer = max_log_likelihood_tracer
        self.max_log_likelihood_fit = max_log_likelihood_fit
        self.max_log_likelihood_mesh_grids_of_planes = (
            max_log_likelihood_mesh_grids_of_planes
        )
        self.pixelization = pixelization
        self.positions = positions
        self.updated_positions = (
            updated_positions if updated_positions is not None else []
        )
        self.updated_positions_threshold = updated_positions_threshold
        self._stochastic_log_likelihoods = stochastic_log_likelihoods

    @property
    def last(self):
        return self

    def stochastic_log_likelihoods(self):
        return self._stochastic_log_likelihoods

    @property
    def image_plane_multiple_image_positions_of_source_plane_centres(self):
        return self.updated_positions
