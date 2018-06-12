from auto_lens.analysis import non_linear
from auto_lens.analysis import model_mapper as mm
from auto_lens.analysis import analyse
from auto_lens.imaging import grids


class ModelAnalysis(object):
    def __init__(self, image, mask, lens_galaxy_priors, source_galaxy_priors, pixelization,
                 model_mapper=mm.ModelMapper(), non_linear_optimizer=non_linear.MultiNestWrapper(),
                 likelihood_for_tracer=analyse.likelihood_for_tracer):
        """
        A class encapsulating an analysis. An analysis takes an image and a set of galaxy priors describing an
        assumed model and applies a pixelization and non linear optimizer to find the best possible fit between the
        image and model.

        Parameters
        ----------
        image: Image
            An image of the galaxy to be fit
        mask: Mask
            A mask eliminating the areas of the image that are not relevant to the analysis
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
        self.mask = mask
        self.lens_galaxy_priors = lens_galaxy_priors
        self.source_galaxy_priors = source_galaxy_priors
        self.pixelization = pixelization
        self.non_linear_optimizer = non_linear_optimizer
        self.model_mapper = model_mapper
        self.likelihood_for_tracer = likelihood_for_tracer

        # TODO: is this step correct? How is the mask chosen?
        self.image_grid_collection = grids.GridCoordsCollection.from_mask(mask)

        for galaxy_prior in lens_galaxy_priors + source_galaxy_priors:
            galaxy_prior.attach_to_model_mapper(model_mapper)

        self.lens_galaxies = None
        self.source_galaxies = None
        self.likelihood = None

    def fitness_function(self, physical_values):
        # Recover classes from physical values
        model_instance = self.model_mapper.from_physical_vector(physical_values)
        # Construct galaxies from their priors
        self.lens_galaxies = list(map(lambda galaxy_prior: galaxy_prior.galaxy_for_model_instance(model_instance),
                                      self.lens_galaxy_priors))
        self.source_galaxies = list(map(lambda galaxy_prior: galaxy_prior.galaxy_for_model_instance(model_instance),
                                        self.source_galaxy_priors))
        # TODO: Construct ray tracing here
        # tracer = ray_tracing.Tracer(self.image, self.lens_galaxies, self.source_galaxies)
        # Determine likelihood:
        self.likelihood = self.likelihood_for_tracer(None)
        return self.likelihood

    def run(self):
        self.non_linear_optimizer.run(self.fitness_function, self.model_mapper.priors_ordered_by_id)
        return ModelAnalysis.Result(self.likelihood, self.lens_galaxies, self.source_galaxies)

    class Result(object):
        def __init__(self, likelihood, lens_galaxies, source_galaxies):
            self.likelihood = likelihood
            self.lens_galaxies = lens_galaxies
            self.source_galaxies = source_galaxies


class LinearPipeline(object):
    def __init__(self, *analyses):
        self.analyses = analyses

    def run(self):
        results = []
        for analysis in self.analyses:
            results.append(analysis.run())
        return results
