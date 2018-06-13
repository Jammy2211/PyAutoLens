from auto_lens.analysis import non_linear
from auto_lens.analysis import model_mapper as mm
from auto_lens.analysis import fitting
from auto_lens.imaging import grids
from auto_lens.analysis import ray_tracing

# TODO: Maybe we need a general Analysis class where each argument is either a model or model instance and any model
# TODO: is a constant and any model instance a variable? The distinction between fixed and variable objects would be
# TODO: made between the constructor and run function.


attribute_map = {"pixelization_class": "pixelization",
                 "instrumentation_class": "instrumentation",
                 "lens_galaxy_priors": "lens_galaxies",
                 "source_galaxy_priors": "source_galaxies"}


class Analysis(object):
    def __init__(self, model_mapper=mm.ModelMapper(), non_linear_optimizer=non_linear.MultiNestWrapper(), **kwargs):
        self.model_mapper = model_mapper
        self.non_linear_optimizer = non_linear_optimizer
        self.included_attributes = []

        for key, value in kwargs.items():
            setattr(self, key, value)
            self.included_attributes.append(key)

        self.missing_attributes = [value for key, value in attribute_map.items() if key not in self.included_attributes]

        # for galaxy_prior in lens_galaxy_priors + source_galaxy_priors:
        #     galaxy_prior.attach_to_model_mapper(model_mapper)

        # model_mapper.add_class('pixelization', pixelization_class)
        # model_mapper.add_class('instrumentation', instrumentation_class)

        def add_galaxy_priors(name):
            if hasattr(self, name):
                for galaxy_prior in getattr(self, name):
                    galaxy_prior.attach_to_model_mapper(model_mapper)

        add_galaxy_priors('lens_galaxy_priors')
        add_galaxy_priors('source_galaxy_priors')

        def add_class(name):
            attribute_name = "{}_class".format(name)
            if hasattr(self, attribute_name):
                model_mapper.add_class(name, getattr(self, attribute_name))

        add_class('instrumentation')
        add_class('pixelization')

    def run(self, image, mask, **kwargs):
        for attribute in self.missing_attributes:
            if attribute not in kwargs:
                raise AssertionError("{} is required".format(attribute))

        for key in kwargs.keys():
            if key not in self.missing_attributes:
                raise AssertionError("A model has been defined for {}".format(key))

        image_grid_collection = grids.GridCoordsCollection.from_mask(mask)
        run = Analysis.Run(image, image_grid_collection, self.model_mapper)

        kwargs.update(self.__dict__)

        for key, value in kwargs.items():
            setattr(run, key, value)

        self.non_linear_optimizer.run(run.fitness_function, self.model_mapper.priors_ordered_by_id)
        return self.__class__.Result(run)

    class Result(object):
        def __init__(self, run):
            for name in attribute_map.values():
                setattr(self, name, getattr(run, name))

            self.likelihood = run.likelihood

    class Run(object):
        def __init__(self, image, image_grid_collection, model_mapper):
            self.image = image
            self.image_grid_collection = image_grid_collection
            self.model_mapper = model_mapper

        # noinspection PyAttributeOutsideInit
        def fitness_function(self, physical_values):
            model_instance = self.model_mapper.from_physical_vector(physical_values)

            if hasattr(self, "pixelization_class"):
                self.pixelization = model_instance.pixelization
            if hasattr(self, "instrumentation_class"):
                self.instrumentation = model_instance.instrumentation
            if hasattr(self, "lens_galaxy_priors"):
                self.lens_galaxies = list(
                    map(lambda galaxy_prior: galaxy_prior.galaxy_for_model_instance(model_instance),
                        self.lens_galaxy_priors))
            if hasattr(self, "source_galaxy_priors"):
                self.source_galaxies = list(
                    map(lambda galaxy_prior: galaxy_prior.galaxy_for_model_instance(model_instance),
                        self.source_galaxy_priors))

            # Construct a ray tracer
            tracer = ray_tracing.Tracer(self.lens_galaxies, self.source_galaxies, self.image_grid_collection)
            # Determine likelihood:
            self.likelihood = fitting.likelihood_for_image_tracer_pixelization_and_instrumentation(self.image,
                                                                                                   tracer,
                                                                                                   self.pixelization,
                                                                                                   self.instrumentation)
            return self.likelihood


class ModelAnalysis(Analysis):
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

        super().__init__(model_mapper=model_mapper, non_linear_optimizer=non_linear_optimizer,
                         lens_galaxy_priors=lens_galaxy_priors, source_galaxy_priors=source_galaxy_priors)


class HyperparameterAnalysis(Analysis):
    def __init__(self, pixelization_class, instrumentation_class, model_mapper=mm.ModelMapper(),
                 non_linear_optimizer=non_linear.MultiNestWrapper()):
        """
        An analysis to improve hyperparameter settings. This optimizes pixelization and instrumentation.

        Parameters
        ----------
        pixelization_class
        instrumentation_class
        model_mapper
        non_linear_optimizer
        """
        super().__init__(model_mapper, non_linear_optimizer, pixelization_class=pixelization_class,
                         instrumentation_class=instrumentation_class)


class MainPipeline(object):
    def __init__(self, *model_analyses, hyperparameter_analysis):
        """
        The primary pipeline. This pipeline runs a series of model analyses with hyperparameter analyses in between.

        Parameters
        ----------
        model_analyses: [ModelAnalysis]
            A series of analysis, each with a fixed model, pixelization and instrumentation but variable model instance.
        hyperparameter_analysis: HyperparameterAnalysis
            An analysis with a fixed model instance but variable pixelization and instrumentation instances.
        """
        self.model_analyses = model_analyses
        self.hyperparameter_analysis = hyperparameter_analysis

    def run(self, image, mask, pixelization, instrumentation):
        """
        Run this pipeline on an image and mask with a given initial pixelization and instrumentation.

        Parameters
        ----------
        image: Image
            The image to be fit
        mask: Mask
            A mask describing which parts of the image are to be included
        pixelization: Pixelization
            The initial pixelization of the source plane
        instrumentation: Instrumentation
            The initial instrumentation

        Returns
        -------
        results_tuple: ([ModelAnalysis.Result], [HyperparameterAnalysis.Result])
            A tuple with a list of results from each model analysis and a list of results from each hyperparameter
            analysis
        """

        # Define lists to keep results in
        model_results = []
        hyperparameter_results = []

        # Run through each model analysis
        for model_analysis in self.model_analyses:
            # Analyse the model
            model_result = model_analysis.run(image, mask, pixelization, instrumentation)
            # Analyse the hyper parameters
            hyperparameter_result = self.hyperparameter_analysis.run(image, mask,
                                                                     lens_galaxies=model_result.lens_galaxies,
                                                                     source_galaxies=model_result.source_galaxies)

            # Update the hyperparameters
            # noinspection PyUnresolvedReferences
            pixelization = hyperparameter_result.pixelization
            # noinspection PyUnresolvedReferences
            instrumentation = hyperparameter_result.instrumentation

            # Append results for these two analyses
            model_results.append(model_result)
            hyperparameter_results.append(hyperparameter_result)

        return model_results, hyperparameter_results
