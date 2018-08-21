from autolens.analysis import galaxy_prior as gp
from autolens.analysis import galaxy as g
from autolens.analysis import ray_tracing
from autolens.imaging import mask as msk
from autolens.imaging import masked_image as mi
from autolens.imaging import image as img
from autolens.analysis import fitting
from autolens.autopipe import non_linear
from autolens import conf
import numpy as np
import logging
from astropy.io import fits
import matplotlib.pyplot as plt
import os

from autolens.pipeline.phase_property import PhaseProperty

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

    def hyper_run(self, image, previous_results=None):
        raise NotImplementedError()


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

    def run(self, image, previous_results=None):
        """
        Run this phase.

        Parameters
        ----------
        previous_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        image: img.Image
            An masked_image that has been masked

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit model and other data.
        """
        analysis = self.make_analysis(image=image, previous_results=previous_results)

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
                                           visualizer=self.__class__.Visualizer)
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
        def __init__(self, masked_image, phase_name, visual_data, visualizer, previous_results=None):
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

        def visual_data_from_tracer_images(self, analysis, image, blurring_image):
            image = analysis.masked_image.convolver_image.convolve_image(image, blurring_image)
            return self.visual_data_2d_from_1d(analysis, image)

        def visual_data_2d_from_1d(self, analysis, image):
            return analysis.masked_image.map_to_2d(image)

    class Visualizer(object):

        def __init__(self, analysis):

            self.phase_path = "{}/".format(conf.instance.output_path) + '/' + analysis.phase_name
            self.image_path = self.phase_path + '/images/'
            self.final_image_path = self.phase_path + '/final_images/'
            make_path_if_does_not_exist(path=self.image_path)
            make_path_if_does_not_exist(path=self.final_image_path)

            shape_arc_seconds = analysis.masked_image.mask.shape_arc_seconds
            self.xticks = np.linspace(-shape_arc_seconds[1] / 2.0, shape_arc_seconds[1] / 2.0, 4)
            self.yticks = np.linspace(-shape_arc_seconds[0] / 2.0, shape_arc_seconds[0] / 2.0, 4)
            self.plot_count = 0
            self.as_fits_during_analysis = conf.instance.general.get('output', 'visualize_as_fits_during_analysis',
                                                                     bool)
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
            super(Phase.Result, self).__init__(constant, likelihood, variable)

        def galaxy_images_from_plane(self, analysis, images, blurring_images):
            blurred_images = list(map(lambda image, blurring_image:
                                      analysis.masked_image.convolver_image.convolve_image(image, blurring_image),
                                      images, blurring_images))
            return list(map(lambda blurred_image: analysis.masked_image.map_to_2d(blurred_image), blurred_images))

        @property
        def model_image(self):
            return np.sum(np.stack((image for image in self.lens_plane_galaxy_images if image is not None)), axis=0)


class LensProfilePhase(Phase):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhaseProperty("lens_galaxies")

    def __init__(self, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name="lens_only_phase"):
        super().__init__(optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                         mask_function=mask_function, phase_name=phase_name)
        self.lens_galaxies = lens_galaxies

    class Analysis(Phase.Analysis):

        def __init__(self, masked_image, phase_name, visual_data, visualizer, previous_results=None):
            super().__init__(masked_image, phase_name, visual_data, visualizer, previous_results)

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
            self.tracer = analysis.tracer_for_instance(instance)
            fitter = fitting.ProfileFitter(analysis.masked_image, self.tracer)
            self.lens_plane_image = self.visual_data_from_tracer_images(analysis,
                                                                        sum(self.tracer.image_plane.galaxy_images),
                                                                        sum(
                                                                            self.tracer.image_plane.galaxy_blurring_images))
            self.residuals = analysis.masked_image.map_to_2d(fitter.blurred_image_residuals)
            self.chi_squareds = analysis.masked_image.map_to_2d(fitter.blurred_image_chi_squareds)

    class Visualizer(Phase.Visualizer):

        def __init__(self, analysis):
            super().__init__(analysis)

        def output_visual_data_as_pngs(self, visual_data, suffix=None, during_analysis=False):
            super().output_visual_data_as_pngs(visual_data, suffix, during_analysis)
            self.output_array_as_png(visual_data.lens_plane_image, 'lens_plane', 'Lens Plane Image', during_analysis)

        def output_visual_data_as_fits(self, visual_data, suffix=None, during_analysis=False):
            super().output_visual_data_as_fits(visual_data, suffix, during_analysis)
            self.output_array_as_fits(visual_data.lens_plane_image, "lens_plane", suffix, during_analysis)

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super().__init__(constant, likelihood, variable, analysis)
            tracer = analysis.tracer_for_instance(constant)
            self.lens_plane_galaxy_images = self.galaxy_images_from_plane(analysis, tracer.image_plane.galaxy_images,
                                                                          tracer.image_plane.galaxy_blurring_images)

        @property
        def lens_plane_image(self):
            return sum(self.lens_plane_galaxy_images)


class LensProfileHyperPhase(LensProfilePhase):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhaseProperty("lens_galaxies")

    def __init__(self, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name="lens_only_hyper_phase"):
        super().__init__(lens_galaxies=lens_galaxies, optimizer_class=optimizer_class,
                         sub_grid_size=sub_grid_size, mask_function=mask_function,
                         phase_name=phase_name)

    class Analysis(LensProfilePhase.Analysis):

        def __init__(self, masked_image, phase_name, visual_data, visualizer, previous_results=None):
            super().__init__(masked_image, phase_name, visual_data, visualizer, previous_results)
            self.hyper_model_image = self.masked_image.mask.map_to_1d(previous_results.last.lens_plane_image)
            self.hyper_galaxy_images = list(map(self.masked_image.mask.map_to_1d,
                                                previous_results.last.lens_plane_galaxy_images))
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
            fitter = fitting.HyperProfileFitter(self.masked_image, tracer, self.hyper_model_image,
                                                self.hyper_galaxy_images, self.hyper_minimum_values)
            return fitter.blurred_image_scaled_likelihood

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens analysis for... \n\nHyper Lens Galaxy::\n{}\n\n".format(instance.lens_galaxies))

    class VisualData(LensProfilePhase.VisualData):

        def __init__(self, instance, analysis):
            super().__init__(instance, analysis)
            hyper_fitter = fitting.HyperProfileFitter(analysis.masked_image, self.tracer, analysis.hyper_model_image,
                                                      analysis.hyper_galaxy_images, analysis.hyper_minimum_values)
            self.scaled_noise = analysis.masked_image.map_to_2d(hyper_fitter.scaled_noise)
            self.scaled_chi_squareds = analysis.masked_image.map_to_2d(hyper_fitter.blurred_image_scaled_chi_squareds)

    class Visualizer(LensProfilePhase.Visualizer):

        def __init__(self, analysis):
            super().__init__(analysis)

        def output_visual_data_as_pngs(self, visual_data, suffix=None, during_analysis=True):
            super().output_visual_data_as_pngs(visual_data, suffix, during_analysis)
            self.output_array_as_png(visual_data.scaled_noise, 'scaled_noise', 'Scaled Noise', during_analysis)
            self.output_array_as_png(visual_data.scaled_chi_squareds, 'scaled_chi_squareds', 'Scaled Noise',
                                     during_analysis)

        def output_visual_data_as_fitss(self, visual_data, suffix=None, during_analysis=True):
            super().output_visual_data_as_fits(visual_data, suffix, during_analysis)
            self.output_array_as_fits(visual_data.scaled_noise, "scaled_noise", suffix, during_analysis)
            self.output_array_as_fits(visual_data.scaled_chi_squareds, "scaled_chi_squareds", suffix, during_analysis)

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super().__init__(constant, likelihood, variable, analysis)


class LensLightHyperOnlyPhase(LensProfileHyperPhase, HyperOnly):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhaseProperty("lens_galaxies")

    def __init__(self, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name="lens_only_hyper_phase", hyper_index=None):
        super().__init__(lens_galaxies=lens_galaxies, optimizer_class=optimizer_class,
                         sub_grid_size=sub_grid_size, mask_function=mask_function,
                         phase_name=phase_name)
        self.hyper_index = hyper_index

    def hyper_run(self, image, previous_results=None):
        class LensGalaxyHyperPhase(LensLightHyperOnlyPhase):
            def pass_priors(self, previous_results):
                use_hyper_galaxy = len(previous_results[-1].constant.lens_galaxies) * [None]
                use_hyper_galaxy[self.hyper_index] = g.HyperGalaxy

                self.lens_galaxies = list(map(lambda lens_galaxy, use_hyper:
                                              gp.GalaxyPrior.from_galaxy(lens_galaxy, hyper_galaxy=use_hyper),
                                              previous_results.last.constant.lens_galaxies, use_hyper_galaxy))

        overall_result = previous_results[-1]

        for i in range(len(previous_results[-1].constant.lens_galaxies)):
            phase = LensGalaxyHyperPhase(optimizer_class=non_linear.MultiNest, sub_grid_size=self.sub_grid_size,
                                         mask_function=self.mask_function,
                                         phase_name=self.phase_name + '/lens_gal_' + str(i), hyper_index=i)

            phase.optimizer.n_live_points = 20
            phase.optimizer.sampling_efficiency = 0.8
            result = phase.run(image, previous_results)
            overall_result.constant += result.constant
            overall_result.variable.lens_galaxies[i].hyper_galaxy = result.variable.lens_galaxies[i].hyper_galaxy

        return overall_result

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
                                           visualizer=self.__class__.Visualizer, hyper_index=self.hyper_index)
        return analysis

    class Analysis(LensProfileHyperPhase.Analysis):

        def __init__(self, masked_image, phase_name, visual_data, visualizer, previous_results=None, hyper_index=None):
            super().__init__(masked_image, phase_name, visual_data, visualizer, previous_results)

            self.hyper_model_image = self.masked_image.mask.map_to_1d(previous_results.last.lens_plane_image)
            hyper_galaxy_image = previous_results.last.lens_plane_galaxy_images[hyper_index]
            self.hyper_galaxy_images = [self.masked_image.mask.map_to_1d(hyper_galaxy_image)]
            self.hyper_minimum_values = len(self.hyper_galaxy_images) * [0.0]


class LensMassAndSourceProfilePhase(Phase):
    """
    Fit a simple source and lens system.
    """

    lens_galaxies = PhaseProperty("lens_galaxies")
    source_galaxies = PhaseProperty("source_galaxies")

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

        super(LensMassAndSourceProfilePhase, self).__init__(optimizer_class=optimizer_class,
                                                            sub_grid_size=sub_grid_size, mask_function=mask_function,
                                                            phase_name=phase_name)
        self.lens_galaxies = lens_galaxies
        self.source_galaxies = source_galaxies

    class Analysis(Phase.Analysis):

        def __init__(self, masked_image, phase_name, visual_data, visualizer, previous_results=None):
            super().__init__(masked_image, phase_name, visual_data, visualizer,
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
                "\nRunning lens/source analysis for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                    instance.lens_galaxies, instance.source_galaxies))

        def tracer_for_instance(self, instance):
            return ray_tracing.Tracer(instance.lens_galaxies, instance.source_galaxies, self.masked_image.grids)

    class VisualData(Phase.VisualData):

        def __init__(self, instance, analysis):
            self.tracer = analysis.tracer_for_instance(instance)
            fitter = fitting.ProfileFitter(analysis.masked_image, self.tracer)
            self.source_plane_image = self.visual_data_from_tracer_images(analysis,
                                                                          sum(self.tracer.source_plane.galaxy_images),
                                                                          sum(
                                                                              self.tracer.source_plane.galaxy_blurring_images))
            self.residuals = analysis.masked_image.map_to_2d(fitter.blurred_image_residuals)
            self.chi_squareds = analysis.masked_image.map_to_2d(fitter.blurred_image_chi_squareds)

    class Visualizer(Phase.Visualizer):

        def __init__(self, analysis):
            super().__init__(analysis)

        def output_visual_data_as_pngs(self, visual_data, suffix=None, during_analysis=False):
            super().output_visual_data_as_pngs(visual_data, suffix,
                                               during_analysis)
            self.output_array_as_png(visual_data.source_plane_image, 'source_plane', 'Source Plane Image',
                                     during_analysis)

        def output_visual_data_as_fits(self, visual_data, suffix=None, during_analysis=False):
            super().output_visual_data_as_fits(visual_data, suffix,
                                               during_analysis)
            self.output_array_as_fits(visual_data.source_plane_image, "source_plane", suffix, during_analysis)

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super().__init__(constant, likelihood, variable, analysis)
            tracer = analysis.tracer_for_instance(constant)
            self.source_plane_galaxy_images = self.galaxy_images_from_plane(analysis, tracer.source_plane.galaxy_images,
                                                                            tracer.source_plane.galaxy_blurring_images)

        @property
        def source_plane_image(self):
            return sum(self.source_plane_galaxy_images)


class LensMassAndSourceProfileHyperPhase(LensMassAndSourceProfilePhase):
    """
    Fit a simple source and lens system.
    """

    lens_galaxies = PhaseProperty("lens_galaxies")
    source_galaxies = PhaseProperty("source_galaxies")

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
        super().__init__(lens_galaxies=lens_galaxies,
                         source_galaxies=source_galaxies, optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                         mask_function=mask_function, phase_name=phase_name)
        self.lens_galaxies = lens_galaxies
        self.source_galaxies = source_galaxies

    class Analysis(LensMassAndSourceProfilePhase.Analysis):

        def __init__(self, masked_image, phase_name, visual_data, visualizer, previous_results=None):
            super().__init__(masked_image, phase_name, visual_data, visualizer, previous_results)
            self.hyper_model_image = self.masked_image.mask.map_to_1d(previous_results.last.source_plane_image)
            self.hyper_galaxy_images = list(map(self.masked_image.mask.map_to_1d,
                                                previous_results.last.source_plane_galaxy_images))
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
            tracer = self.tracer_for_instance(instance)
            fitter = fitting.HyperProfileFitter(self.masked_image, tracer, self.hyper_model_image,
                                                self.hyper_galaxy_images, self.hyper_minimum_values)
            return fitter.blurred_image_scaled_likelihood

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens/source analysis for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                    instance.lens_galaxies, instance.source_galaxies))

    class VisualData(LensMassAndSourceProfilePhase.VisualData):

        def __init__(self, instance, analysis):
            super().__init__(instance, analysis)
            hyper_fitter = fitting.HyperProfileFitter(analysis.masked_image, self.tracer, analysis.hyper_model_image,
                                                      analysis.hyper_galaxy_images, analysis.hyper_minimum_values)
            self.scaled_noise = analysis.masked_image.map_to_2d(hyper_fitter.scaled_noise)
            self.scaled_chi_squareds = analysis.masked_image.map_to_2d(hyper_fitter.blurred_image_scaled_chi_squareds)

    class Visualizer(LensMassAndSourceProfilePhase.Visualizer):

        def __init__(self, analysis):
            super().__init__(analysis)

        def output_visual_data_as_pngs(self, visual_data, suffix=None, during_analysis=False):
            super().output_visual_data_as_pngs(visual_data, suffix,
                                               during_analysis)
            self.output_array_as_png(visual_data.scaled_noise, 'scaled_noise', 'Scaled Noise', during_analysis)
            self.output_array_as_png(visual_data.scaled_chi_squareds, 'scaled_chi_squareds', 'Scaled Noise',
                                     during_analysis)

        def output_visual_data_as_fits(self, visual_data, suffix=None, during_analysis=False):
            super().output_visual_data_as_fits(visual_data, suffix,
                                               during_analysis)
            self.output_array_as_fits(visual_data.scaled_noise, "scaled_noise", suffix, during_analysis)
            self.output_array_as_fits(visual_data.scaled_chi_squareds, "scaled_chi_squareds", suffix, during_analysis)

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super().__init__(constant, likelihood, variable, analysis)


class LensMassAndSourceProfileHyperOnlyPhase(LensMassAndSourceProfileHyperPhase, HyperOnly):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhaseProperty("lens_galaxies")
    source_galaxies = PhaseProperty("source_galaxies")

    def __init__(self, source_galaxies=None, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name="source_and_len_hyper_phase", hyper_index=None):
        super().__init__(lens_galaxies=lens_galaxies,
                         source_galaxies=source_galaxies, optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                         mask_function=mask_function, phase_name=phase_name)
        self.hyper_index = hyper_index

    def hyper_run(self, image, previous_results=None):
        class SourceGalaxyHyperPhase(LensMassAndSourceProfileHyperOnlyPhase):
            def pass_priors(self, previous_results):
                use_hyper_galaxy = len(previous_results[-1].constant.source_galaxies) * [None]
                use_hyper_galaxy[self.hyper_index] = g.HyperGalaxy

                self.lens_galaxies = previous_results[-1].variable.lens_galaxies
                self.lens_galaxies[0].sie = previous_results[0].constant.lens_galaxies[0].sie
                self.source_galaxies = list(map(lambda source_galaxy, use_hyper:
                                                gp.GalaxyPrior.from_galaxy(source_galaxy, hyper_galaxy=use_hyper),
                                                previous_results.last.constant.source_galaxies, use_hyper_galaxy))

        overall_result = previous_results[-1]

        for i in range(len(previous_results[-1].constant.source_galaxies)):
            phase = SourceGalaxyHyperPhase(optimizer_class=non_linear.MultiNest, sub_grid_size=self.sub_grid_size,
                                           mask_function=self.mask_function,
                                           phase_name=self.phase_name + '/src_gal_' + str(i), hyper_index=i)

            phase.optimizer.n_live_points = 20
            phase.optimizer.sampling_efficiency = 0.8
            result = phase.run(image, previous_results)
            overall_result.constant += result.constant
            overall_result.variable.source_galaxies[i].hyper_galaxy = result.variable.source_galaxies[i].hyper_galaxy

        return overall_result

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
                                           visualizer=self.__class__.Visualizer, hyper_index=self.hyper_index)
        return analysis

    class Analysis(LensMassAndSourceProfileHyperPhase.Analysis):

        def __init__(self, masked_image, phase_name, visual_data, visualizer, previous_results=None, hyper_index=None):
            super().__init__(masked_image, phase_name, visual_data, visualizer, previous_results)

            self.hyper_model_image = self.masked_image.mask.map_to_1d(previous_results.last.source_plane_image)
            hyper_galaxy_image = previous_results.last.source_plane_galaxy_images[hyper_index]
            self.hyper_galaxy_images = [self.masked_image.mask.map_to_1d(hyper_galaxy_image)]
            self.hyper_minimum_values = len(self.hyper_galaxy_images) * [0.0]


def make_path_if_does_not_exist(path):
    if os.path.exists(path) == False:
        os.makedirs(path)
