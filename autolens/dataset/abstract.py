from autolens import exc
from autolens.fit import fit


class AbstractLensMasked:
    def __init__(
        self, inversion_stochastic, positions_threshold, preload_sparse_grids_of_planes
    ):

        self.inversion_stochastic = inversion_stochastic
        self.positions_threshold = positions_threshold

        self.preload_sparse_grids_of_planes = preload_sparse_grids_of_planes

    def check_positions_trace_within_threshold_via_tracer(self, tracer):

        if not tracer.has_mass_profile or len(tracer.planes) == 1:
            return

        if self.positions is not None and self.positions_threshold is not None:

            positions_fit = fit.FitPositions(
                positions=self.positions,
                tracer=tracer,
                noise_map=self.imaging.pixel_scales,
            )

            if not positions_fit.maximum_separation_within_threshold(
                self.positions_threshold
            ):
                raise exc.RayTracingException

    def check_inversion_pixels_are_below_limit_via_tracer(self, tracer):

        if self.inversion_pixel_limit is not None:
            pixelizations = list(filter(None, tracer.pixelizations_of_planes))
            if pixelizations:
                for pixelization in pixelizations:
                    if pixelization.pixels > self.inversion_pixel_limit:
                        raise exc.PixelizationException
