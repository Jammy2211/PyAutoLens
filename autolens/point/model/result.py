import autoarray as aa

from autolens.analysis.result import Result


class ResultPoint(Result):
    @property
    def grid(self):
        return aa.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

    @property
    def max_log_likelihood_fit(self):
        return self.analysis.fit_from(instance=self.instance)

    def _max_log_likelihood_component_fit(self):
        """
        Returns the component fit (positions, flux or time-delay) of the maximum log likelihood `FitPointDataset`
        which carries the per-image magnification accessors.

        Every component fit of a `FitPointDataset` is an `AbstractFitPoint` evaluated at the observed positions
        of the same `PointDataset` with the same tracer and point-source name, so they all give identical
        magnifications; the first present component (positions, then flux, then time delays) is used.
        """
        fit = self.max_log_likelihood_fit

        for component in ("positions", "flux", "time_delays"):
            component_fit = getattr(fit, component, None)
            if component_fit is not None and hasattr(
                component_fit, "signed_magnifications_at_positions"
            ):
                return component_fit

        raise AttributeError(
            f"The maximum log likelihood fit of this ResultPoint is of type {type(fit).__name__}, which has no "
            f"positions, flux or time-delay component fit carrying `signed_magnifications_at_positions`, so the "
            f"per-image magnifications cannot be computed from it."
        )

    @property
    def signed_magnifications_at_positions(self) -> aa.ArrayIrregular:
        """
        The signed magnification at every observed image position of the point-source dataset, computed from the
        maximum log likelihood tracer via the Hessian. The sign is the physical image parity (negative for
        saddle-point images); see `AbstractFitPoint.signed_magnifications_at_positions`.
        """
        return (
            self._max_log_likelihood_component_fit().signed_magnifications_at_positions
        )

    @property
    def magnifications_at_positions(self) -> aa.ArrayIrregular:
        """
        The unsigned magnification |mu| at every observed image position of the point-source dataset, computed from
        the maximum log likelihood tracer via the Hessian; see `AbstractFitPoint.magnifications_at_positions`.
        """
        return self._max_log_likelihood_component_fit().magnifications_at_positions
