import autoarray as aa

from autolens.imaging.fit_imaging import FitImaging
from autolens.analysis.result import ResultDataset
from autolens.analysis.preloads import Preloads


class ResultImaging(ResultDataset):
    """
    After the non-linear search of a fit to an imaging dataset is complete it creates this `ResultImaging`, object
    which includes:

    - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
    the maximum likelihood model, posteriors and other properties.

    - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
    an instance of the maximum log likelihood model).

    - The non-linear search used to perform the model fit.

    This class contains a number of methods which use the above objects to create the max log likelihood `Plane`,
    `FitImaging`, hyper-galaxy images,etc.

    Parameters
    ----------
    samples
        A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
        run of samples of the nested sampler.
    model
        The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
        the imaging data.
    search
        The non-linear search used to perform this model-fit.

    Returns
    -------
    ResultImaging
        The result of fitting the model to the imaging dataset, via a non-linear search.
    """

    @property
    def max_log_likelihood_fit(self) -> FitImaging:
        """
        An instance of a `FitImaging` corresponding to the maximum log likelihood model inferred by the non-linear
        search.
        """
        return self.analysis.fit_imaging_via_instance_from(
            instance=self.instance,
            preload_overwrite=Preloads(use_w_tilde=False),
            check_positions=False,
        )

    @property
    def unmasked_model_image(self) -> aa.Array2D:
        """
        The model image of the maximum log likelihood model, created without using a mask.
        """
        return self.max_log_likelihood_fit.unmasked_blurred_image

    @property
    def unmasked_model_image_of_planes(self):
        """
        A list of the model image of every plane in the maximum log likelihood model, where all images are created
        without using a mask.
        """
        return self.max_log_likelihood_fit.unmasked_blurred_image_of_planes_list
