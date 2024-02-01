import numpy as np

import autoarray as aa
import autogalaxy as ag

from autogalaxy.analysis.adapt_images import AdaptImages
from autolens.lens.ray_tracing import Tracer
from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.analysis.result import ResultDataset


class ResultInterferometer(ResultDataset):
    """
    After the non-linear search of a fit to an interferometer dataset is complete it creates
    this `ResultInterferometer` object, which includes:

    - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
      the maximum likelihood model, posteriors and other properties.

    - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
      an instance of the maximum log likelihood model).

    - The non-linear search used to perform the model fit.

    This class contains a number of methods which use the above objects to create the max log likelihood `Tracer`,
    `FitInterferometer`, adapt-galaxy images,etc.

    Parameters
    ----------
    samples
        A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
        run of samples of the nested sampler.
    model
        The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
        the interferometer data.
    search
        The non-linear search used to perform this model-fit.

    Returns
    -------
    ResultInterferometer
        The result of fitting the model to the interferometer dataset, via a non-linear search.
    """

    @property
    def max_log_likelihood_fit(self) -> FitInterferometer:
        """
        An instance of a `FitInterferometer` corresponding to the maximum log likelihood model inferred by the
        non-linear search.
        """
        return self.analysis.fit_from(instance=self.instance)

    @property
    def max_log_likelihood_tracer(self) -> Tracer:
        """
        An instance of a `Tracer` corresponding to the maximum log likelihood model inferred by the non-linear search.

        The `Tracer` is computed from the `max_log_likelihood_fit`, as this ensures that all linear light profiles
        are converted to normal light profiles with their `intensity` values updated.
        """
        return (
            self.max_log_likelihood_fit.model_obj_linear_light_profiles_to_light_profiles
        )

    @property
    def real_space_mask(self) -> aa.Mask2D:
        """
        The real space mask used by this model-fit.
        """
        return self.max_log_likelihood_fit.dataset.real_space_mask

    def adapt_images_from(self, use_model_images : bool = False) -> AdaptImages:
        """
        Returns the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
        reconstructed galaxy's morphology.

        This can use either:

        - The model image of each galaxy in the best-fit model.
        - The subtracted image of each galaxy in the best-fit model, where the subtracted image is the dataset
          minus the model images of all other galaxies.

        In **PyAutoLens** these adapt images have had lensing calculations performed on them and therefore for source
        galaxies are their lensed model images in the image-plane.

        Parameters
        ----------
        use_model_images
            If True, the model images of the galaxies are used to create the adapt images. If False, the subtracted
            images of the galaxies are used.
        """

        return AdaptImages.from_result(
            result=self,
            use_model_images=True
        )