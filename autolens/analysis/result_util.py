import numpy as np

from autoarray.structures.grids.two_d import grid_2d_irregular
from autolens.fit import fit_point_source


def updated_positions_from(positions, results, auto_positions_factor):
    """
    If automatic position updating is on, update the phase's positions using the results of the previous phase's
    lens model, by ray-tracing backwards the best-fit source centre(s) to the image-plane.

    The outcome of this function are as follows:

    1) If auto positioning is off (self.auto_positions_factor is None), use the previous phase's positions.
    2) If auto positioning is on (self.auto_positions_factor not None) use positions based on the previous phase's
       best-fit tracer. However, if this tracer gives 1 or less positions, use the previous positions.
    3) If this previous tracer is composed of only 1 plane (e.g. you are light profile fitting the image-plane
       only), use the previous positions.
    4) If auto positioning is on or off and there is no previous phase, use the input positions.
    """

    if results is None:
        return positions

    if results.last is not None:

        if not hasattr(results.last, "positions"):
            return positions
        try:
            if len(results.last.max_log_likelihood_tracer.planes) <= 1:
                return positions
        except AttributeError:
            pass

    if auto_positions_factor is not None and results.last is not None:

        updated_positions = (
            results.last.image_plane_multiple_image_positions_of_source_plane_centres
        )

        # TODO : Coorrdinates refascotr will sort out index call here

        if isinstance(updated_positions, grid_2d_irregular.Grid2DIrregular):
            if len(updated_positions.in_list) > 1:
                return updated_positions

    if results.last is not None:
        if results.last.positions is not None:
            return results.last.positions

    return positions


def updated_positions_threshold_from(
    positions,
    results,
    positions_threshold,
    auto_positions_factor,
    auto_positions_minimum_threshold,
) -> [float]:
    """
    If automatic position updating is on, update the phase's threshold using this phase's updated positions.

    First, we ray-trace forward the positions of the source-plane centres (see above) via the mass model to
    determine how far apart they are separated. This gives us their source-plane sepration, which is multiplied by
    self.auto_positions_factor to set the threshold.

    The threshold is rounded up to the auto positions minimum threshold if that setting is included."""

    if results is None:
        return positions_threshold

    if results.last is not None:

        try:
            if len(results.last.max_log_likelihood_tracer.planes) <= 1:
                if positions_threshold is not None:
                    return positions_threshold
                if auto_positions_minimum_threshold is not None:
                    return auto_positions_minimum_threshold
                return None
        except AttributeError:
            pass

    if auto_positions_factor and results.last is not None:

        if positions is None:
            return None

        positions_fits = fit_point_source.FitPositionsSourceMaxSeparation(
            positions=positions,
            noise_map=None,
            tracer=results.last.max_log_likelihood_tracer,
        )

        new_positions_threshold = auto_positions_factor * np.max(
            positions_fits.max_separation_of_source_plane_positions
        )

    else:

        new_positions_threshold = positions_threshold

    if auto_positions_minimum_threshold is not None and positions_threshold is not None:
        if (positions_threshold < auto_positions_minimum_threshold) or (
            positions_threshold is None
        ):
            new_positions_threshold = auto_positions_minimum_threshold

    return new_positions_threshold
