from autolens import exc
from autolens.fit import fit_positions


class AbstractLensMasked:
    def __init__(self, positions_threshold, preload_sparse_grids_of_planes):

        self.positions_threshold = positions_threshold

        self.preload_sparse_grids_of_planes = preload_sparse_grids_of_planes

    def check_positions_trace_within_threshold_via_tracer(self, tracer):

        if not tracer.has_mass_profile or len(tracer.planes) == 1:
            return

        if self.positions is not None and self.positions_threshold is not None:

            positions_fit = fit_positions.FitPositionsSourcePlaneMaxSeparation(
                positions=self.positions,
                tracer=tracer,
                noise_value=self.imaging.pixel_scales,
            )

            if not positions_fit.maximum_separation_within_threshold(
                self.positions_threshold
            ):
                raise exc.RayTracingException
