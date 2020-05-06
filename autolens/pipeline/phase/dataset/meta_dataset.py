import autoarray as aa
from autolens import exc
from autolens.fit import fit

import numpy as np


class MetaLens:
    def __init__(
        self,
        auto_positions_factor=None,
        positions_threshold=None,
        pixel_scale_interpolation_grid=None,
        inversion_uses_border=True,
    ):

        self.auto_positions_factor = auto_positions_factor
        self.positions_threshold = positions_threshold
        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid
        self.inversion_uses_border = inversion_uses_border

    def updated_positions_from_positions_and_results(self, positions, results):
        """If automatic position updating is on, update the phase's positions using the results of the previous phase's
        lens model, by ray-tracing backwards the best-fit source centre(s) to the image-plane.

        The outcome of this function are as follows:

        1) If auto positioning is off (self.auto_positions_factor is None), use the previous phase's positions.
        2) If auto positioning is on (self.auto_positions_factor not None) use positions based on the previous phase's
           best-fit tracer. However, if this tracer gives 1 or less positions, use the previous positions.
        3) If auto positioning is on or off and there is no previous phase, use the input positions.
        """

        if results.last is not None:
            if not hasattr(results.last, "positions"):
                return positions

        if self.auto_positions_factor is not None and results.last is not None:

            updated_positions = (
                results.last.image_plane_multiple_image_positions_of_source_plane_centres
            )

            # TODO : Coorrdinates refascotr will sort out index call here

            if isinstance(updated_positions, aa.Coordinates):
                if updated_positions.in_list:
                    if len(updated_positions.in_list[0]) > 1:
                        return updated_positions

        if results.last is not None:
            if results.last.positions is not None:
                if results.last.positions.in_list:
                    return results.last.positions

        return positions

    def updated_positions_threshold_from_positions(self, positions, results) -> [float]:
        """
        If automatic position updating is on, update the phase's threshold using this phase's updated positions.

        First, we ray-trace forward the positions of the source-plane centres (see above) via the mass model to
        determine how far apart they are separated. This gives us their source-plane sepration, which is multiplied by
        self.auto_positions_factor to set the threshold."""

        if self.auto_positions_factor and results.last is not None:

            if positions is None:
                return None

            positions_fits = fit.FitPositions(
                positions=aa.Coordinates(coordinates=positions),
                tracer=results.last.max_log_likelihood_tracer,
                noise_map=1.0,
            )

            return self.auto_positions_factor * np.max(
                positions_fits.maximum_separations
            )

        else:

            return self.positions_threshold

    def check_positions(self, positions):

        if self.positions_threshold is not None and positions is None:
            raise exc.PhaseException(
                "You have specified for a phase to use positions, but not input positions to the "
                "pipeline when you ran it."
            )

    def preload_pixelization_grids_of_planes_from_results(self, results):

        if self.is_hyper_phase:
            return None

        if (
            results.last is not None
            and self.pixelization is not None
            and not self.pixelizaition_is_model
        ):
            if self.pixelization.__class__ is results.last.pixelization.__class__:
                if hasattr(results.last, "hyper_combined"):
                    return (
                        results.last.hyper_combined.max_log_likelihood_pixelization_grids_of_planes
                    )
                else:
                    return results.last.max_log_likelihood_pixelization_grids_of_planes
        return None
