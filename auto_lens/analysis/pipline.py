class Analysis(object):
    def __init__(self, image, lens_galaxy_priors, source_galaxy_priors, pixelization, non_linear_optimizer):
        """
        A class encapsulating an analysis. An analysis takes an image and a set of galaxy priors describing an
        assumed model and applies a pixelization and non linear optimizer to find the best possible fit between the
        image and model.

        Parameters
        ----------
        image: Image
            An image of the galaxy to be fit
        lens_galaxy_priors: [GalaxyPrior]
            A list of prior instances describing the lens
        source_galaxy_priors: [GalaxyPrior]
            A list of prior instances describing the source
        pixelization: Pixelization
            A class determining how the source plane should be pixelized
        non_linear_optimizer: NonLinearOptimizer
            A wrapper around a library that searches an n-dimensional space by providing unit vector hypercubes to
            the analysis.
        """

        self.image = image
        self.lens_galaxy_priors = lens_galaxy_priors
        self.source_galaxy_priors = source_galaxy_priors
        self.pixelization = pixelization
        self.non_linear_optimizer = non_linear_optimizer
        pass
