from auto_lens.analysis import non_linear
from auto_lens.analysis import model_mapper as mm
from auto_lens.analysis import analyse
from auto_lens.imaging import grids
from auto_lens.analysis import ray_tracing


class ModelStage(object):
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
        """
        Generates a model instance from a set of physical values, uses this to construct galaxies and ultimately a
        tracer that can be passed to analyse.likelihood_for_tracer.

        Parameters
        ----------
        physical_values: [float]
            Physical values from a non-linear optimiser

        Returns
        -------
        likelihood: float
            A value for the likelihood associated with this set of physical values in conjunction with the provided
            model
        """

        # Recover classes from physical values
        model_instance = self.model_mapper.from_physical_vector(physical_values)
        # Construct galaxies from their priors
        self.lens_galaxies = list(map(lambda galaxy_prior: galaxy_prior.galaxy_for_model_instance(model_instance),
                                      self.lens_galaxy_priors))
        self.source_galaxies = list(map(lambda galaxy_prior: galaxy_prior.galaxy_for_model_instance(model_instance),
                                        self.source_galaxy_priors))
        # Construct a ray tracer
        tracer = ray_tracing.Tracer(self.lens_galaxies, self.source_galaxies, self.image_grid_collection)
        # Determine likelihood:
        self.likelihood = self.likelihood_for_tracer(tracer)
        return self.likelihood

    def run(self):
        """
        Run the analysis, iteratively analysing each set of values provided by the non-linear optimiser until it
        terminates.

        Returns
        -------
        result: Result
            The result of the analysis, comprising a likelihood and the final set of galaxies generated
        """

        # Run the optimiser using the fitness function. The fitness function is applied iteratively until the optimiser
        # terminates
        self.non_linear_optimizer.run(self.fitness_function, self.model_mapper.priors_ordered_by_id)
        return self.__class__.Result(self)

    class Result(object):
        def __init__(self, analysis):
            """
            The result of an analysis

            Parameters
            ----------
            analysis: ModelStage
                An analysis
            """
            # The final likelihood found
            self.likelihood = analysis.likelihood
            # The final lens galaxies generated
            self.lens_galaxies = analysis.lens_galaxies
            # The final source galaxies generated
            self.source_galaxies = analysis.source_galaxies


class LinearPipeline(object):
    def __init__(self, *stages):
        """
        A pipeline to linearly apply a series of analyses

        Parameters
        ----------
        stages: Stage...
            A series of analyses to be applied in order
        """
        self.stages = stages

    def run(self):
        """
        Run the pipeline

        Returns
        -------
        results: Results
            A list of result objects describing the results of the analyses
        """
        results = []
        for stage in self.stages:
            results.append(stage.run())
        return results
