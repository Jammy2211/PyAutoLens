from autolens.analysis import galaxy_prior as gp
from autolens.analysis import galaxy as g
from autolens.analysis import ray_tracing
from autolens.imaging import mask as msk
from autolens.imaging import masked_image as mi
from autolens.imaging import image as img
from autolens.analysis import fitting
from autolens.autopipe import non_linear
from autolens.autopipe import model_mapper as mm
from autolens import conf
import numpy as np
from autolens.pixelization import pixelization as px
import inspect
import logging
from astropy.io import fits
import matplotlib.pyplot as plt
import shutil
import os

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def default_mask_function(image):
    return msk.Mask.circular(image.shape_arc_seconds, image.pixel_scale, 3)


class ResultsCollection(list):
    def __init__(self, results):
        super().__init__(results)

    @property
    def last(self):
        if len(self) > 0:
            return self[-1]
        return None

    @property
    def first(self):
        if len(self) > 0:
            return self[0]
        return None


class IntervalCounter(object):
    def __init__(self, interval):
        self.count = 0
        self.interval = interval

    def __call__(self):
        self.count += 1
        return self.count % self.interval == 0


class HyperOnly(object):
    pass


class Phase(object):
    def __init__(self, optimizer_class=non_linear.MultiNest, sub_grid_size=1, mask_function=default_mask_function,
                 phase_name=None):
        """
        A phase in an analysis pipeline. Uses the set non_linear optimizer to try to fit models and images passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        self.optimizer = optimizer_class(name=phase_name)
        self.sub_grid_size = sub_grid_size
        self.mask_function = mask_function
        self.phase_name = phase_name
        self.hyper_index = None

    @property
    def constant(self):
        """
        Convenience method

        Returns
        -------
        ModelInstance
            A model instance comprising all the constant objects in this analysis
        """
        return self.optimizer.constant

    @property
    def variable(self):
        """
        Convenience method

        Returns
        -------
        ModelMapper
            A model mapper comprising all the variable (prior) objects in this analysis
        """
        return self.optimizer.variable

    @property
    def doc(self):
        if self.__doc__ is not None:
            return self.__doc__.replace("  ", "").replace("\n", " ")

    def run(self, image, last_results=None):
        """
        Run this phase.

        Parameters
        ----------
        last_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        image: img.Image
            An masked_image that has been masked

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit model and other data.
        """
        analysis = self.make_analysis(image=image, previous_results=last_results)

        result = self.optimizer.fit(analysis)
        visual_data = analysis.visual_data(result.constant, analysis)
        analysis.visualizer.output_visual_data_as_pngs(visual_data, 'best_fit', during_analysis=False)
        analysis.visualizer.output_visual_data_as_fits(visual_data, 'best_fit', during_analysis=False)
        return self.__class__.Result(result.constant, result.likelihood, result.variable, analysis)

    def make_analysis(self, image, previous_results=None):
        """
        Create an analysis object. Also calls the prior passing and masked_image modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        image: im.Image
            An masked_image that has been masked
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """
        mask = self.mask_function(image)
        image = self.modify_image(image, previous_results)
        masked_image = mi.MaskedImage(image, mask, sub_grid_size=self.sub_grid_size)
        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(masked_image=masked_image, phase_name=self.phase_name,
                                           previous_results=previous_results,
                                           visual_data=self.__class__.VisualData,
                                           visualizer=self.__class__.Visualizer,
                                           hyper_index=self.hyper_index)
        return analysis

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def modify_image(self, image, previous_results):
        """
        Customize an masked_image. e.g. removing lens light.

        Parameters
        ----------
        image: img.Image
            An masked_image that has been masked
        previous_results: ResultsCollection
            The result of the previous analysis

        Returns
        -------
        masked_image: img.Image
            The modified masked_image (not changed by default)
        """
        return image

    def pass_priors(self, previous_results):
        """
        Perform any prior or constant passing. This could involve setting model attributes equal to priors or constants
        from a previous phase.

        Parameters
        ----------
        previous_results: ResultsCollection
            The result of the previous phase
        """
        pass

    class Analysis(object):
        def __init__(self, masked_image, phase_name, visual_data, visualizer, previous_results=None, hyper_index=None):
            """
            An analysis object

            Parameters
            ----------
            previous_results: ResultsCollection
                The results of all previous phases
            masked_image: mi.MaskedImage
                An masked_image that has been masked
            """
            self.previous_results = previous_results
            self.phase_name = phase_name
            self.masked_image = masked_image
            log_interval = conf.instance.general.get('output', 'log_interval', int)
            self.__should_log = IntervalCounter(log_interval)
            visualise_interval = conf.instance.general.get('output', 'visualise_interval', int)
            self.__should_visualise = IntervalCounter(visualise_interval)
            self.visual_data = visual_data
            self.visualizer = visualizer(self)

        @property
        def should_log(self):
            return self.__should_log()

        @property
        def should_visualise(self):
            return self.__should_visualise()

        @property
        def last_results(self):
            if self.previous_results is not None:
                return self.previous_results.last

        def fit(self, instance):
            raise NotImplementedError()

        def try_visualize(self, instance):
            """
            Determine the fitness of a particular model

            Parameters
            ----------
            instance
                A model instance

            Returns
            -------
            fit: fitting.Fit
                How fit the model is and the model
            """
            if self.should_log:
                self.log(instance)
            if self.should_visualise:
                self.visualizer.plot_count += 1
                logger.info("Saving visualisations {}".format(self.visualizer.plot_count))
                visual_data = self.visual_data(instance, self)
                self.visualizer.output_visual_data_as_fits(visual_data, during_analysis=True)
                self.visualizer.output_visual_data_as_pngs(visual_data, during_analysis=True)
            return None

        @classmethod
        def log(cls, instance):
            raise NotImplementedError()

        def tracer_for_instance(self, instance):
            raise NotImplementedError()


    class VisualData(object):

        def __init__(self, instance, analysis):

            self.tracer = analysis.tracer_for_instance(instance)
            self.fitter = fitting.ProfileFitter(analysis.masked_image, self.tracer)
            self.residuals = analysis.masked_image.map_to_2d(self.fitter.blurred_image_residuals)
            self.chi_squareds = analysis.masked_image.map_to_2d(self.fitter.blurred_image_chi_squareds)


    class Visualizer(object):

        def __init__(self, analysis):

            self.phase_path = "{}/".format(conf.instance.output_path) + '/' + analysis.phase_name
            self.image_path = self.phase_path + '/images/'
            self.final_image_path = self.phase_path + '/final_images/'
            make_path_if_does_not_exist(path=self.image_path)
            make_path_if_does_not_exist(path=self.final_image_path)

            shape_arc_seconds = analysis.masked_image.mask.shape_arc_seconds
            self.xticks = np.linspace(-shape_arc_seconds[1]/2.0, shape_arc_seconds[1]/2.0, 4)
            self.yticks = np.linspace(-shape_arc_seconds[0]/2.0, shape_arc_seconds[0]/2.0, 4)
            self.plot_count = 0
            self.as_fits_during_analysis = conf.instance.general.get('output', 'visualize_as_fits_during_analysis', bool)
            self.as_fits_at_end = conf.instance.general.get('output', 'visualize_as_fits_at_end', bool)

            self.total_plots = 2

            self.output_array_as_png(analysis.masked_image.image, 'observed_image', 'Observed Image', True)

        def output_visual_data_as_pngs(self, visual_data, suffix=None, during_analysis=True):

            self.output_array_as_png(visual_data.residuals, 'residuals', 'Image Residuals', during_analysis)
            self.output_array_as_png(visual_data.chi_squareds, 'chi_squareds', 'Chi Squareds', during_analysis)

        def output_visual_data_as_fits(self, visual_data, suffix=None, during_analysis=True):

            self.output_array_as_fits(visual_data.residuals, "residuals", suffix, during_analysis)
            self.output_array_as_fits(visual_data.chi_squareds, "chi_squareds", suffix, during_analysis)

        def output_array_as_png(self, array, filename, title, during_analysis):

            file = self.file_path_and_name(filename, '.png', during_analysis)

            if os.path.isfile(file):
                os.remove(file)

            plt.figure(figsize=(28, 20))
            plt.xticks(array.shape[0] * np.array([0.0, 0.33, 0.66, 0.99]), self.xticks)
            plt.yticks(array.shape[1] * np.array([0.0, 0.33, 0.66, 0.99]), self.yticks)
            plt.tick_params(labelsize=30)
            plt.imshow(array, aspect='auto')
            plt.title(title, fontsize=32)
            plt.xlabel('x (arcsec)', fontsize=36)
            plt.ylabel('y (arcsec)', fontsize=36)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=28)
            plt.savefig(file, bbox_inches='tight')
            plt.close()

        def output_array_as_fits(self, array, filename, suffix, during_analysis):

            if (during_analysis is True and self.as_fits_during_analysis is True) or during_analysis is False:

                file = self.file_path_and_name(filename, '.fits', during_analysis)

                if os.path.isfile(file):
                    os.remove(file)

                try:
                    if array is not None:
                        hdu = fits.PrimaryHDU()
                        hdu.data = array
                        hdu.writeto(file)
                except OSError as e:
                    logger.exception(e)

        def file_path_and_name(self, filename, extension, during_analysis):

            if during_analysis is True:
                return self.image_path + str(self.plot_count) + '_' + filename + extension
            elif during_analysis is False:
                return self.final_image_path + filename + extension


    class Result(non_linear.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            self.lens_plane_galaxy_images = []
            super(Phase.Result, self).__init__(constant, likelihood, variable)

        @property
        def model_image(self):
            return np.sum(np.stack((image for image in self.lens_plane_galaxy_images if image is not None)), axis=0)


def phase_property(name):
    """
    Create a property that is tied to the non_linear instance determines whether to set itself as a constant or
    variable.

    Parameters
    ----------
    name: str
        The phase_name of this variable

    Returns
    -------
    property: property
        A property that appears to be an attribute of the phase but is really an attribute of constant or variable.
    """

    def fget(self):
        if hasattr(self.optimizer.constant, name):
            return getattr(self.optimizer.constant, name)
        elif hasattr(self.optimizer.variable, name):
            return getattr(self.optimizer.variable, name)

    def fset(self, value):
        if inspect.isclass(value) or isinstance(value, gp.GalaxyPrior) or isinstance(value, list):
            setattr(self.optimizer.variable, name, value)
            try:
                delattr(self.optimizer.constant, name)
            except AttributeError:
                pass
        else:
            setattr(self.optimizer.constant, name, value)
            try:
                delattr(self.optimizer.variable, name)
            except AttributeError:
                pass

    return property(fget=fget, fset=fset, doc=name)


class LensPlanePhase(Phase):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = phase_property("lens_galaxies")

    def __init__(self, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name="lens_only_phase"):
        super(LensPlanePhase, self).__init__(optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                                             mask_function=mask_function, phase_name=phase_name)
        self.lens_galaxies = lens_galaxies

    class Analysis(Phase.Analysis):

        def __init__(self, masked_image, phase_name, visualizer, visual_data, previous_results=None, hyper_index=None):

            super(LensPlanePhase.Analysis, self).__init__(masked_image, phase_name, visual_data, visualizer,
                                                          previous_results)

        def fit(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the masked_image in this analysis.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model masked_image itself
            """
            self.try_visualize(instance)
            tracer = self.tracer_for_instance(instance)
            fitter = fitting.ProfileFitter(self.masked_image, tracer)
            return fitter.blurred_image_likelihood

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens analysis for... \n\nLens Galaxy::\n{}\n\n".format(instance.lens_galaxies))

        def tracer_for_instance(self, instance):
            return ray_tracing.Tracer(instance.lens_galaxies, [], self.masked_image.grids)


    class VisualData(Phase.VisualData):

        def __init__(self, instance, analysis):

            super(LensPlanePhase.VisualData, self).__init__(instance, analysis)
            self.lens_plane_image = analysis.masked_image.map_to_2d(sum(self.tracer.image_plane.galaxy_images))


    class Visualizer(Phase.Visualizer):

        def __init__(self, analysis):

            super(LensPlanePhase.Visualizer, self).__init__(analysis)

        def output_visual_data_as_pngs(self, visual_data, suffix=None, during_analysis=False):

            super(LensPlanePhase.Visualizer, self).output_visual_data_as_pngs(visual_data, suffix, during_analysis)
            self.output_array_as_png(visual_data.lens_plane_image, 'lens_plane', 'Lens Plane Image', during_analysis)

        def output_visual_data_as_fits(self, visual_data, suffix=None, during_analysis=False):

            super(LensPlanePhase.Visualizer, self).output_visual_data_as_fits(visual_data, suffix, during_analysis)
            self.output_array_as_fits(visual_data.lens_plane_image, "lens_plane", suffix, during_analysis)


    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super(LensPlanePhase.Result, self).__init__(constant, likelihood, variable, analysis)
            tracer = analysis.tracer_for_instance(constant)
            self.lens_plane_galaxy_images = list(map(lambda galaxy_image : analysis.masked_image.mask.map_to_2d(galaxy_image),
                                                     tracer.image_plane.galaxy_images))

        @property
        def lens_plane_image(self):
            return sum(self.lens_plane_galaxy_images)


class LensPlaneHyperPhase(LensPlanePhase):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = phase_property("lens_galaxies")

    def __init__(self, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name="lens_only_hyper_phase", hyper_index=None):

        super(LensPlaneHyperPhase, self).__init__(lens_galaxies=lens_galaxies, optimizer_class=optimizer_class,
                                                  sub_grid_size=sub_grid_size, mask_function=mask_function,
                                                  phase_name=phase_name)
        self.hyper_index = hyper_index

    class Analysis(LensPlanePhase.Analysis):

        def __init__(self, masked_image, phase_name, visualizer, visual_data, previous_results=None, hyper_index=None):

            super(LensPlaneHyperPhase.Analysis, self).__init__(masked_image, phase_name, visualizer, visual_data,
                                                               previous_results)
            self.hyper_model_image = self.masked_image.mask.map_to_1d(previous_results.last.lens_plane_image)
            if hyper_index is None:
                self.hyper_galaxy_images = list(map(self.masked_image.mask.map_to_1d, 
                                                previous_results.last.lens_plane_galaxy_images))
            else:
                hyper_galaxy_image = previous_results.last.lens_plane_galaxy_images[hyper_index]
                self.hyper_galaxy_images = [self.masked_image.mask.map_to_1d(hyper_galaxy_image)]
            self.hyper_minimum_values = len(self.hyper_galaxy_images) * [0.0]

        def fit(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the masked_image in this analysis.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model masked_image itself
            """
            self.try_visualize(instance)
            tracer = self.tracer_for_instance(instance)
            fitter = fitting.HyperProfileFitter(self.masked_image, tracer,  self.hyper_model_image,
                                                self.hyper_galaxy_images, self.hyper_minimum_values)
            return fitter.blurred_image_scaled_likelihood

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens analysis for... \n\nHyper Lens Galaxy::\n{}\n\n".format(instance.lens_galaxies))


    class VisualData(LensPlanePhase.VisualData):

        def __init__(self, instance, analysis):

            super(LensPlaneHyperPhase.VisualData, self).__init__(instance, analysis)
            self.hyper_fitter = fitting.HyperProfileFitter(analysis.masked_image, self.tracer, analysis.hyper_model_image,
                                                analysis.hyper_galaxy_images, analysis.hyper_minimum_values)
            self.scaled_noise = analysis.masked_image.map_to_2d(self.hyper_fitter.scaled_noise)
            self.scaled_chi_squareds = analysis.masked_image.map_to_2d(self.hyper_fitter.blurred_image_scaled_chi_squareds)


    class Visualizer(LensPlanePhase.Visualizer):

        def __init__(self, analysis):

            super(LensPlaneHyperPhase.Visualizer, self).__init__(analysis)

        def output_visual_data_as_pngs(self, visual_data, suffix=None, during_analysis=True):

            super(LensPlanePhase.Visualizer, self).output_visual_data_as_pngs(visual_data, suffix, during_analysis)
            self.output_array_as_png(visual_data.scaled_noise, 'scaled_noise', 'Scaled Noise', during_analysis)
            self.output_array_as_png(visual_data.scaled_chi_squareds, 'scaled_chi_squareds', 'Scaled Noise', during_analysis)

        def output_visual_data_as_fitss(self, visual_data, suffix=None, during_analysis=True):

            super(LensPlanePhase.Visualizer, self).output_visual_data_as_fits(visual_data, suffix, during_analysis)
            self.output_array_as_fits(visual_data.scaled_noise, "scaled_noise", suffix, during_analysis)
            self.output_array_as_fits(visual_data.scaled_chi_squareds, "scaled_chi_squareds", suffix, during_analysis)


    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super(LensPlaneHyperPhase.Result, self).__init__(constant, likelihood, variable, analysis)


class LensPlaneHyperOnlyPhase(LensPlaneHyperPhase, HyperOnly):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = phase_property("lens_galaxies")

    def __init__(self, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name="lens_only_hyper_phase"):
        super(LensPlaneHyperOnlyPhase, self).__init__(lens_galaxies=lens_galaxies, optimizer_class=optimizer_class,
                                                      sub_grid_size=sub_grid_size, mask_function=mask_function,
                                                      phase_name=phase_name)

    def run(self, image, last_results=None):

        class LensGalaxyHyperPhase(LensPlaneHyperPhase):
            def pass_priors(self, previous_results):

                use_hyper_galaxy = len(previous_results[-1].constant.lens_galaxies)*[None]
                use_hyper_galaxy[self.hyper_index] = g.HyperGalaxy

                self.lens_galaxies = list(map(lambda lens_galaxy, use_hyper :
                                              gp.GalaxyPrior.from_galaxy(lens_galaxy, hyper_galaxy=use_hyper),
                                              previous_results.last.constant.lens_galaxies, use_hyper_galaxy))

        overall_result = last_results[-1]

        for i in range(len(last_results[-1].constant.lens_galaxies)):

            phase = LensGalaxyHyperPhase(optimizer_class=non_linear.MultiNest, sub_grid_size=self.sub_grid_size,
                                         mask_function=self.mask_function,
                                         phase_name=self.phase_name+'/lens_galaxy_'+str(i), hyper_index=i)

            phase.optimizer.n_live_points = 20
            phase.optimizer.sampling_efficiency = 0.8

            result = phase.run(image, last_results)
            overall_result.constant += result.constant
            overall_result.variable.lens_galaxies[i].hyper_galaxy = result.variable.lens_galaxies[i].hyper_galaxy

        return self.__class__.Result(overall_result.constant, overall_result.likelihood, overall_result.variable,
                                     self.make_analysis(image, last_results))


class LensSourcePhase(LensPlanePhase):
    """
    Fit a simple source and lens system.
    """

    source_galaxies = phase_property("source_galaxies")

    def __init__(self, lens_galaxies=None, source_galaxies=None, optimizer_class=non_linear.DownhillSimplex,
                 sub_grid_size=1, mask_function=default_mask_function, phase_name="source_lens_phase"):
        """
        A phase with a simple source/lens model

        Parameters
        ----------
        lens_galaxies : [g.Galaxy] | [gp.GalaxyPrior]
            A galaxy that acts as a gravitational lens
        source_galaxies: [g.Galaxy] | [gp.GalaxyPrior]
            A galaxy that is being lensed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super().__init__(lens_galaxies=lens_galaxies, optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                         mask_function=mask_function, phase_name=phase_name)
        self.source_galaxies = source_galaxies

    def modify_image(self, image, last_result):
        """

        Parameters
        ----------
        image: im.Image
        last_result: LensSourcePhase.Result

        Returns
        -------

        """
        return image

    class Analysis(Phase.Analysis):

        def __init__(self, masked_image, phase_name, previous_results=None):

            super(LensSourcePhase.Analysis, self).__init__(masked_image, phase_name, previous_results)

            self.hyper_model_image = None
            self.hyper_galaxy_images = None
            if self.last_results is not None:
                self.hyper_model_image = self.masked_image.mask.map_to_1d(previous_results.last.model_image)
                self.hyper_galaxy_images = list(
                    map(self.masked_image.mask.map_to_1d, previous_results.last.galaxy_images))
                # TODO : We currently get these from tracer using defaults. Lets now set them up here via a config.
                # TODO : This is just a placehold for now
                self.hyper_minimum_values = len(self.hyper_galaxy_images) * [0.02]

        def fit(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the masked_image in this analysis.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model masked_image itself
            """
            super(LensSourcePhase.Analysis, self).fit(instance)

            tracer = self.tracer_for_instance(instance)

            if self.last_results is None or not tracer.all_with_hyper_galaxies:

                fitter = fitting.ProfileFitter(self.masked_image, tracer)
                return fitter.blurred_image_likelihood

            elif self.last_results is not None and tracer.all_with_hyper_galaxies:

                fitter = fitting.HyperProfileFitter(self.masked_image, tracer, self.hyper_model_image,
                                                    self.hyper_galaxy_images, self.hyper_minimum_values)
                return fitter.blurred_image_scaled_likelihood

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens/source analysis for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                    instance.lens_galaxies, instance.source_galaxies))

        def tracer_for_instance(self, instance):
            return ray_tracing.Tracer(instance.lens_galaxies, instance.source_galaxies, self.masked_image.grids)


    class Visualizer(Phase.Visualizer):

        def __init__(self, analysis):

            super(LensSourcePhase.Visualizer, self).__init__(analysis)

        def output_visual_data_as_pngs(self, instance, analysis, suffix=None):

            tracer = analysis.tracer_for_instance(instance)
            fitter = fitting.ProfileFitter(analysis.masked_image, tracer)

            # lens_plane_image = analysis.masked_image.map_to_2d(sum(tracer.image_plane.lens_plane_galaxy_images))
            # self.output_lens_plane_image(array=lens_plane_image)
            # self.output_as_fits(array=lens_plane_image, filename="/lens_plane_image", suffix=suffix)
            #
            # source_plane_image = analysis.masked_image.map_to_2d(sum(tracer.source_plane.lens_plane_galaxy_images)
            # self.output_source_plane_image(array=source_plane_image)
            # self.output_as_fits(array=source_plane_image, filename="/source_plane_image", suffix=suffix)

            residuals = analysis.masked_image.map_to_2d(fitter.blurred_image_residuals)
            self.output_residuals(array=residuals)
            self.output_array_as_fits(array=residuals, filename="/residuals", suffix=suffix)

            chi_squareds = analysis.masked_image.map_to_2d(fitter.blurred_image_chi_squareds)
            self.output_chi_squareds(array=chi_squareds)
            self.output_array_as_fits(array=chi_squareds, filename="/chi_squareds", suffix=suffix)

        def output_source_plane_image(self, array):

            self.output_array_as_png(array, '/source_plane', '.png', 'Lens Plane Image')


    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super(LensSourcePhase.Result, self).__init__(constant, likelihood, variable, analysis)
            tracer = analysis.tracer_for_instance(constant)
            self.galaxy_images = [analysis.masked_image.mask.map_to_2d(tracer.image_plane.galaxy_images),
                                  analysis.masked_image.mask.map_to_2d(tracer.image_plane.galaxy_images)]

        @property
        def lens_galaxy_image(self):
            return self.galaxy_images[0]

        @property
        def source_galaxy_image(self):
            return self.galaxy_images[1]


class SourceLensHyperGalaxyPhase(LensSourcePhase):
    """
    Adjust hyper galaxy parameters to optimize the fit.
    """

    def run(self, image, last_results=None):
        class LensPhase(LensSourcePhase):
            def pass_priors(self, previous_results):
                self.lens_galaxy = gp.GalaxyPrior.from_galaxy(
                    previous_results.last.constant.lens_galaxy,
                    hyper_galaxy=g.HyperGalaxy)
                self.source_galaxy = previous_results.last.constant.source_galaxy

        class SourcePhase(LensSourcePhase):
            def pass_priors(self, previous_results):
                self.lens_galaxy = previous_results.last.constant.lens_galaxy
                self.source_galaxy = gp.GalaxyPrior.from_galaxy(
                    previous_results.last.constant.source_galaxy,
                    hyper_galaxy=g.HyperGalaxy)

        lens_result = LensPhase(phase_name="{}_lens".format(self.phase_name)).run(image, last_results)
        source_result = SourcePhase(phase_name="{}_lens".format(self.phase_name)).run(image, last_results)

        return self.__class__.Result(lens_result.constant + source_result.constant,
                                     (lens_result.likelihood + source_result.likelihood) / 2,
                                     lens_result.variable + source_result.variable,
                                     self.make_analysis(image, last_results))


class PixelizedSourceLensPhase(LensSourcePhase):
    """
    Fit a simple source and lens system using a pixelized source.
    """

    def __init__(self, lens_galaxy=None, pixelization=px.RectangularRegConst,
                 optimizer_class=non_linear.DownhillSimplex, sub_grid_size=1, mask_function=default_mask_function):
        super().__init__(lens_galaxy=lens_galaxy, source_galaxies=gp.GalaxyPrior(pixelization=pixelization),
                         optimizer_class=optimizer_class, sub_grid_size=sub_grid_size, mask_function=mask_function)

    class Analysis(LensSourcePhase.Analysis):

        def fit(self, lens_galaxy=None, source_galaxy=None):
            """
            Determine the fit of a lens galaxy and source galaxy to the masked_image in this analysis.

            Parameters
            ----------
            lens_galaxy: g.Galaxy
                The galaxy that acts as a gravitational lens
            source_galaxy: g.Galaxy
                The galaxy that produces the light that is being lensed

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model masked_image itself
            """
            if self.should_log:
                logger.debug(
                    "\nRunning lens/source analysis for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                        lens_galaxy,
                        source_galaxy))

            tracer = ray_tracing.Tracer([lens_galaxy], [source_galaxy], self.masked_image.grids)

            # TODO : Don't need a sparse mask for Rectangular pixelization. We'll remove it soon so just ignore it for
            # TODO : now.
            sparse_mask = None

            # TODO : I guess there is overhead doing this, and once we do it once in a fit we dont need to do it again.
            # TODO : Memoize or set up in class constructor?

            if tracer.has_galaxy_with_light_profile and not tracer.has_galaxy_with_pixelization:

                fitter = fitting.ProfileFitter(self.masked_image, tracer)
                return fitter.blurred_image_likelihood

            elif not tracer.has_galaxy_with_light_profile and tracer.has_galaxy_with_pixelization:

                fitter = fitting.PixelizationFitter(self.masked_image, sparse_mask, tracer)
                return fitter.reconstructed_image_evidence

            elif tracer.has_galaxy_with_light_profile and tracer.has_galaxy_with_pixelization:

                fitter = fitting.ProfileFitter(self.masked_image, tracer)
                pix_fitter = fitter.pixelization_fitter_with_profile_subtracted_masked_image(sparse_mask)
                return pix_fitter.reconstructed_image_evidence


def make_path_if_does_not_exist(path):
    if os.path.exists(path) == False:
        os.makedirs(path)