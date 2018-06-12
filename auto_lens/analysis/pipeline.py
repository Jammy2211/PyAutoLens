from auto_lens.analysis import non_linear
from auto_lens.analysis import model_mapper as mm
from auto_lens.analysis import fitting
from auto_lens.imaging import grids
from auto_lens.analysis import ray_tracing


class ModelAnalysis(object):
    def __init__(self, lens_galaxy_priors, source_galaxy_priors, model_mapper=mm.ModelMapper(),
                 non_linear_optimizer=non_linear.MultiNestWrapper()):
        """
        A class encapsulating an analysis. An analysis takes an image and a set of galaxy priors describing an
        assumed model and applies a pixelization and non linear optimizer to find the best possible fit between the
        image and model.

        Parameters
        ----------
        lens_galaxy_priors: [GalaxyPrior]
            A list of prior instances describing the lens
        source_galaxy_priors: [GalaxyPrior]
            A list of prior instances describing the source
        model_mapper: ModelMapper
            A class used to bridge between non linear unit vectors and class instances
        non_linear_optimizer: NonLinearOptimizer
            A wrapper around a library that searches an n-dimensional space by providing unit vector hypercubes to
            the analysis.
        """

        self.lens_galaxy_priors = lens_galaxy_priors
        self.source_galaxy_priors = source_galaxy_priors
        self.non_linear_optimizer = non_linear_optimizer
        self.model_mapper = model_mapper

        for galaxy_prior in lens_galaxy_priors + source_galaxy_priors:
            galaxy_prior.attach_to_model_mapper(model_mapper)

    def run(self, image, mask, pixelization, instrumentation):
        """
        Run the analysis, iteratively analysing each set of values provided by the non-linear optimiser until it
        terminates.

        Parameters
        ----------
        image: Image
            An image
        mask: Mask
            A mask defining which regions of the image should be ignored
        pixelization: Pixelization
            An object determining how the source plane should be pixelized
        instrumentation: Instrumentation
            An object describing instrumental effects to be applied to generated images

        Returns
        -------
        result: Result
            The result of the analysis, comprising a likelihood and the final set of galaxies generated
        """

        # Set up expensive objects that can be used repeatedly for fittings
        image_grid_collection = grids.GridCoordsCollection.from_mask(mask)

        # TODO: Convert image to 1D and create other auxiliary data structures such as kernel makers

        # Create an instance of run that will be used for this analysis
        run = self.__class__.Run(image, image_grid_collection, self.model_mapper, self.lens_galaxy_priors,
                                 self.source_galaxy_priors, pixelization, instrumentation)

        # Run the optimiser using the fitness function. The fitness function is applied iteratively until the optimiser
        # terminates
        self.non_linear_optimizer.run(run.fitness_function, self.model_mapper.priors_ordered_by_id)
        return self.__class__.Result(run)

    class Run(object):
        def __init__(self, image, image_grid_collection, model_mapper, lens_galaxy_priors, source_galaxy_priors,
                     pixelization, instrumentation):
            self.image = image
            self.model_mapper = model_mapper
            self.lens_galaxy_priors = lens_galaxy_priors
            self.source_galaxy_priors = source_galaxy_priors
            self.lens_galaxies = []
            self.source_galaxies = []
            self.image_grid_collection = image_grid_collection
            self.pixelization = pixelization
            self.instrumentation = instrumentation
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
            self.likelihood = fitting.likelihood_for_image_tracer_pixelization_and_instrumentation(self.image, tracer,
                                                                                                   self.pixelization,
                                                                                                   self.instrumentation)
            return self.likelihood

    class Result(object):
        def __init__(self, run):
            """
            The result of an analysis

            Parameters
            ----------
            run: Run
                An analysis run
            """
            # The final likelihood found
            self.likelihood = run.likelihood
            # The final lens galaxies generated
            self.lens_galaxies = run.lens_galaxies
            # The final source galaxies generated
            self.source_galaxies = run.source_galaxies


class HyperparameterAnalysis(object):
    def __init__(self, pixelization_class, instrumentation_class, model_mapper=mm.ModelMapper(),
                 non_linear_optimizer=non_linear.MultiNestWrapper()):
        self.model_mapper = model_mapper
        self.pixelization_class = pixelization_class
        self.instrumentation_class = instrumentation_class
        self.non_linear_optimizer = non_linear_optimizer

    def run(self, lens_galaxies, source_galaxies):
        pass


class MainPipeline(object):
    def __init__(self, *analyses, hyperparameter_analysis):
        self.analyses = analyses
        self.hyperparameter_analysis = hyperparameter_analysis

    def run(self, image, mask, pixelization, instrumentation):
        hyperparameter_results = []
        model_results = []
        return hyperparameter_results, model_results
        # TODO: test and implement


class LinearPipeline(object):
    def __init__(self, *stages):
        """
        A pipeline to linearly apply a series of analyses

        Parameters
        ----------
        stages: Stage...
            A series of stages to be applied in order
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
