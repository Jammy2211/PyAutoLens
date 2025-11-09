import autoarray as aa

from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair


class FitPositionsImagePairRepeat(AbstractFitPositionsImagePair):
    """
    Fits the positions of a a point source dataset using a `Tracer` object with an image-plane chi-squared where every
    model position of the point-source is paired with its closest observed position, allowing for repeated pairings of
    the same observed position to model positions.

    The fit performs the following steps:

    1) Determine the source-plane centre of the point source, which could be a free model parameter or computed
       as the barycenter of ray-traced positions in the source-plane, using name pairing (see below).

    2) Determine the image-plane model positions using the `PointSolver` and the source-plane centre of the point
       source (e.g. ray tracing triangles to and from  the image and source planes), including accounting for
       multi-plane ray-tracing.

    3) Pair each model position with the closest observed position, allowing for repeated pairings of the same
       observed position to model positions, to compute the `residual_map`.

    5) Compute the chi-squared of each position as the square of the residual divided by the RMS noise-map value.

    6) Sum the chi-squared values to compute the overall log likelihood of the fit.

    Point source fitting uses name pairing, whereby the `name` of the `Point` object is paired to the name of the
    point source dataset to ensure that point source datasets are fitted to the correct point source.

    This fit object is used in the `FitPointDataset` to perform position based fitting of a `PointDataset`,
    which may also fit other components of the point dataset like fluxes or time delays.

    When performing a `model-fit`via an `AnalysisPoint` object the `figure_of_merit` of this object
    is called and returned in the `log_likelihood_function`.

    Parameters
    ----------
    name
        The name of the point source dataset which is paired to a `Point` profile.
    data
        The positions of the point source in the image-plane which are fitted.
    noise_map
        The noise-map of the positions which are used to compute the log likelihood of the positions.
    tracer
        The tracer of galaxies whose point source profile are used to fit the positions.
    solver
        Solves the lens equation in order to determine the image-plane positions of a point source by ray-tracing
        triangles to and from the source-plane.
    profile
        Manually input the profile of the point source, which is used instead of the one extracted from the
        tracer via name pairing if that profile is not found.
    """

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        residual_map = []

        for position in self.data:
            distances = [
                self.square_distance(model_position, position)
                for model_position in self.model_data.array
            ]
            residual_map.append(self._xp.sqrt(self._xp.min(self._xp.array(distances))))

        return aa.ArrayIrregular(values=self._xp.array(residual_map))
