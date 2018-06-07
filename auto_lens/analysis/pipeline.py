from auto_lens.analysis import non_linear
from auto_lens.analysis import model_mapper as mm


class ModelAnalysis(object):
    def __init__(self, image, lens_galaxy_priors, source_galaxy_priors, pixelization, model_mapper=mm.ModelMapper(),
                 non_linear_optimizer=non_linear.MultiNestWrapper()):
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
        model_mapper: ModelMapper
            A class used to bridge between non linear unit vectors and class instances
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
        self.model_mapper = model_mapper

        for galaxy_prior in lens_galaxy_priors + source_galaxy_priors:
            galaxy_prior.attach_to_model_mapper(model_mapper)
