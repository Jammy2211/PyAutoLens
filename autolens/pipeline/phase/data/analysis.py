from autolens import exc
from autolens.lens import lens_fit
from autolens.pipeline.phase import abstract


class Analysis(abstract.analysis.AbstractAnalysis):
    @property
    def lens_data(self):
        raise NotImplementedError()

    def check_positions_trace_within_threshold_via_tracer(self, tracer):

        if (
                self.lens_data.positions is not None
                and self.lens_data.positions_threshold is not None
        ):

            traced_positions_of_planes = tracer.traced_positions_of_planes_from_positions(
                positions=self.lens_data.positions
            )

            fit = lens_fit.LensPositionFit(
                positions=traced_positions_of_planes[-1],
                noise_map=self.lens_data.pixel_scale,
            )

            if not fit.maximum_separation_within_threshold(
                    self.lens_data.positions_threshold
            ):
                raise exc.RayTracingException

    def check_inversion_pixels_are_below_limit_via_tracer(self, tracer):

        if self.lens_data.inversion_pixel_limit is not None:
            pixelizations = list(filter(None, tracer.pixelizations_of_planes))
            if pixelizations:
                for pixelization in pixelizations:
                    if pixelization.pixels > self.lens_data.inversion_pixel_limit:
                        raise exc.PixelizationException