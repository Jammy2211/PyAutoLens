import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autolens.data.array import mask as msk
from autolens.data.array.util import binning_util
from autolens.model.galaxy import galaxy as g, galaxy_fit, galaxy_data as gd
from autolens.model.galaxy.plotters import galaxy_fit_plotters
from autolens.model.inversion import pixelizations as pix


def default_mask_function(image):
    return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale,
                             radius_arcsec=3.0)


def setup_phase_mask(data, mask, mask_function, inner_mask_radii):
    if mask_function is not None:
        mask = mask_function(image=data.image)
    elif mask is None and mask_function is None:
        mask = default_mask_function(image=data.image)

    if inner_mask_radii is not None:
        inner_mask = msk.Mask.circular(shape=mask.shape, pixel_scale=mask.pixel_scale,
                                       radius_arcsec=inner_mask_radii, invert=True)
        mask = mask + inner_mask

    return mask


class AbstractPhase(af.AbstractPhase):

    def __init__(self, phase_name, phase_tag=None, phase_folders=tuple(), tag_phases=True,
                 optimizer_class=af.MultiNest,
                 cosmology=cosmo.Planck15, auto_link_priors=False):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit
        models and hyper passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        phase_name: str
            The name of this phase
        """

        self.phase_folders = phase_folders

        super().__init__(phase_name=phase_name, phase_tag=phase_tag,
                         phase_folders=phase_folders, tag_phases=tag_phases,
                         optimizer_class=optimizer_class,
                         auto_link_priors=auto_link_priors)

        self.cosmology = cosmology

    @property
    def variable(self):
        """
        Convenience method

        Returns
        -------
        ModelMapper
            A model mapper comprising all the variable (prior) objects in this lens
        """
        return self.optimizer.variable

    @property
    def phase_property_collections(self):
        """
        Returns
        -------
        phase_property_collections: [PhaseProperty]
            A list of phase property collections associated with this phase. This is
            used in automated prior passing and should be overridden for any phase that
            contains its own PhasePropertys.
        """
        return []

    @property
    def path(self):
        return self.optimizer.path

    @property
    def doc(self):
        if self.__doc__ is not None:
            return self.__doc__.replace("  ", "").replace("\n", " ")

    def pass_priors(self, results):
        """
        Perform any prior or constant passing. This could involve setting model
        attributes equal to priors or constants from a previous phase.

        Parameters
        ----------
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous phase
        """
        pass

    # noinspection PyAbstractClass
    class Analysis(af.Analysis):

        def __init__(self, cosmology, results=None):
            """
            An lens object

            Parameters
            ----------
            results: autofit.tools.pipeline.ResultsCollection
                The results of all previous phases
            """

            self.results = results
            self.cosmology = cosmology

            self.plot_count = 0

        @property
        def last_results(self):
            """
            Returns
            -------
            result: AbstractPhase.Result | None
                The result from the last phase
            """
            if self.results is not None:
                return self.results.last

        def tracer_for_instance(self, instance):
            raise NotImplementedError()

        def padded_tracer_for_instance(self, instance):
            raise NotImplementedError()

        def fit_for_tracers(self, tracer, padded_tracer):
            raise NotImplementedError()

        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

    def make_result(self, result, analysis):
        return self.__class__.Result(constant=result.constant,
                                     figure_of_merit=result.figure_of_merit,
                                     previous_variable=result.previous_variable,
                                     gaussian_tuples=result.gaussian_tuples,
                                     analysis=analysis, optimizer=self.optimizer)

    class Result(af.Result):

        def __init__(self, constant, figure_of_merit, previous_variable,
                     gaussian_tuples, analysis, optimizer):
            """
            The result of a phase
            """
            super(Phase.Result, self).__init__(constant=constant,
                                               figure_of_merit=figure_of_merit,
                                               previous_variable=previous_variable,
                                               gaussian_tuples=gaussian_tuples)

            self.analysis = analysis
            self.optimizer = optimizer

        @property
        def most_likely_tracer(self):
            return self.analysis.tracer_for_instance(instance=self.constant)

        @property
        def most_likely_padded_tracer(self):
            return self.analysis.padded_tracer_for_instance(instance=self.constant)

        @property
        def most_likely_fit(self):
            return self.analysis.fit_for_tracers(
                tracer=self.most_likely_tracer,
                padded_tracer=self.most_likely_padded_tracer)

        @property
        def most_likely_image_plane_pixelization_grid(self):

            if (self.most_likely_tracer.image_plane.grid_stack.pixelization == np.array([[0.0, 0.0]])).all():
                return None
            else:
                return self.most_likely_tracer.image_plane.grid_stack.pixelization

        @property
        def unmasked_model_image(self):
            return self.most_likely_fit.unmasked_blurred_image_plane_image

        @property
        def unmasked_model_image_of_planes(self):
            return self.most_likely_fit.unmasked_blurred_image_plane_image_of_planes

        @property
        def unmasked_model_image_of_planes_and_galaxies(self):
            fit = self.most_likely_fit
            return fit.unmasked_blurred_image_plane_image_of_planes_and_galaxies

        def image_2d_for_galaxy(self, galaxy: g.Galaxy) -> np.ndarray:
            """
            Parameters
            ----------
            galaxy
                A galaxy used in this phase

            Returns
            -------
            ndarray or None
                A numpy array giving the model image of that galaxy
            """
            return self.most_likely_fit.galaxy_image_2d_dict[galaxy]

        @property
        def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
            """
            Tuples associating the names of galaxies with instances from the best fit
            """
            return self.constant.path_instance_tuples_for_class(cls=g.Galaxy)

        @property
        def image_2d_dict(self) -> {str: g.Galaxy}:
            """
            A dictionary associating galaxy names with model images of those galaxies
            """
            return {galaxy_path: self.image_2d_for_galaxy(galaxy) for
                    galaxy_path, galaxy in self.path_galaxy_tuples}

        def image_1d_dict_from_mask(self, mask) -> {str: g.Galaxy}:
            """
            A dictionary associating galaxy names with model images of those galaxies
            """

            image_1d_dict = {}

            for galaxy, galaxy_image_2d in self.image_2d_dict.items():

                image_1d_dict[galaxy] = mask.array_1d_from_array_2d(array_2d=galaxy_image_2d)

            return image_1d_dict

        def hyper_galaxy_image_1d_path_dict_from_mask(self, mask):
            """
            A dictionary associating 1D hyper galaxy images with their names.
            """

            hyper_minimum_percent = \
                af.conf.instance.general.get('hyper', 'hyper_minimum_percent', float)

            image_galaxy_1d_dict = self.image_1d_dict_from_mask(mask=mask)

            hyper_galaxy_image_1d_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:

                galaxy_image_1d = image_galaxy_1d_dict[path]

                minimum_galaxy_value = hyper_minimum_percent * max(galaxy_image_1d)
                galaxy_image_1d[galaxy_image_1d < minimum_galaxy_value] = minimum_galaxy_value
                hyper_galaxy_image_1d_path_dict[path] = galaxy_image_1d

            return hyper_galaxy_image_1d_path_dict

        def hyper_galaxy_image_2d_path_dict_from_mask(self, mask):
            """
            A dictionary associating 2D hyper galaxy images with their names.
            """

            hyper_galaxy_image_1d_path_dict = self.hyper_galaxy_image_1d_path_dict_from_mask(mask=mask)

            hyper_galaxy_image_2d_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:

                hyper_galaxy_image_2d_path_dict[path] = \
                    mask.scaled_array_2d_from_array_1d(array_1d=hyper_galaxy_image_1d_path_dict[path])

            return hyper_galaxy_image_2d_path_dict

        def cluster_image_1d_dict_from_cluster(self, cluster) -> {str: g.Galaxy}:
            """
            A dictionary associating 1D cluster images with their names.
            """

            cluster_image_1d_dict = {}

            for galaxy, galaxy_image_2d in self.image_2d_dict.items():

                cluster_image_2d = binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(
                    array_2d=galaxy_image_2d, bin_up_factor=cluster.bin_up_factor)

                cluster_image_1d_dict[galaxy] = \
                    cluster.mask.array_1d_from_array_2d(array_2d=cluster_image_2d)

            return cluster_image_1d_dict

        def hyper_galaxy_cluster_image_1d_path_dict_from_cluster(self, cluster):
            """
            A dictionary associating 1D hyper galaxy cluster images with their names.
            """

            if cluster is not None:

                hyper_minimum_percent = \
                    af.conf.instance.general.get('hyper', 'hyper_minimum_percent', float)

                cluster_image_1d_galaxy_dict = self.cluster_image_1d_dict_from_cluster(cluster=cluster)

                hyper_galaxy_cluster_image_path_dict = {}

                for path, galaxy in self.path_galaxy_tuples:

                    galaxy_cluster_image_1d = cluster_image_1d_galaxy_dict[path]

                    minimum_cluster_value = hyper_minimum_percent * max(galaxy_cluster_image_1d)
                    galaxy_cluster_image_1d[galaxy_cluster_image_1d < minimum_cluster_value] = minimum_cluster_value

                    hyper_galaxy_cluster_image_path_dict[path] = galaxy_cluster_image_1d

                return hyper_galaxy_cluster_image_path_dict

        def hyper_galaxy_cluster_image_2d_path_dict_from_cluster(
                self, cluster):
            """
            A dictionary associating "D hyper galaxy images cluster images with their names.
            """

            if cluster is not None:

                hyper_galaxy_cluster_image_1d_path_dict = \
                    self.hyper_galaxy_cluster_image_1d_path_dict_from_cluster(cluster=cluster)

                hyper_galaxy_cluster_image_2d_path_dict = {}

                for path, galaxy in self.path_galaxy_tuples:

                    hyper_galaxy_cluster_image_2d_path_dict[path] = \
                        cluster.mask.scaled_array_2d_from_array_1d(array_1d=hyper_galaxy_cluster_image_1d_path_dict[path])

                return hyper_galaxy_cluster_image_2d_path_dict

        def hyper_model_image_1d_from_mask(self, mask):

            hyper_galaxy_image_1d_path_dict = self.hyper_galaxy_image_1d_path_dict_from_mask(mask=mask)

            hyper_model_image_1d = np.zeros(mask.pixels_in_mask)

            for path, galaxy in self.path_galaxy_tuples:
                hyper_model_image_1d += hyper_galaxy_image_1d_path_dict[path]

            return hyper_model_image_1d


class Phase(AbstractPhase):

    def run(self, image, results=None, mask=None):
        raise NotImplementedError()

    # noinspection PyAbstractClass
    class Analysis(AbstractPhase.Analysis):

        def __init__(self, cosmology, results=None):
            super(Phase.Analysis, self).__init__(cosmology=cosmology, results=results)

            self.should_plot_mask = \
                af.conf.instance.visualize.get('figures', 'plot_mask_on_images', bool)
            self.extract_array_from_mask = \
                af.conf.instance.visualize.get('figures', 'extract_images_from_mask', bool)
            self.zoom_around_mask = \
                af.conf.instance.visualize.get('figures', 'zoom_around_mask_of_images', bool)
            self.should_plot_positions = \
                af.conf.instance.visualize.get('figures', 'plot_positions_on_images', bool)
            self.plot_units = \
                af.conf.instance.visualize.get('figures', 'plot_units', str).strip()

            self.plot_ray_tracing_all_at_end_png = \
                af.conf.instance.visualize.get('plots', 'plot_ray_tracing_all_at_end_png',
                                          bool)
            self.plot_ray_tracing_all_at_end_fits = \
                af.conf.instance.visualize.get('plots', 'plot_ray_tracing_all_at_end_fits',
                                          bool)

            self.plot_ray_tracing_as_subplot = \
                af.conf.instance.visualize.get('plots', 'plot_ray_tracing_as_subplot', bool)
            self.plot_ray_tracing_image_plane_image = \
                af.conf.instance.visualize.get('plots',
                                          'plot_ray_tracing_image_plane_image', bool)
            self.plot_ray_tracing_source_plane = \
                af.conf.instance.visualize.get('plots',
                                          'plot_ray_tracing_source_plane_image', bool)
            self.plot_ray_tracing_convergence = \
                af.conf.instance.visualize.get('plots', 'plot_ray_tracing_convergence',
                                          bool)
            self.plot_ray_tracing_potential = \
                af.conf.instance.visualize.get('plots', 'plot_ray_tracing_potential', bool)
            self.plot_ray_tracing_deflections = \
                af.conf.instance.visualize.get('plots', 'plot_ray_tracing_deflections',
                                          bool)


class GalaxyFitPhase(AbstractPhase):
    galaxies = af.PhaseProperty("galaxies")

    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), galaxies=None,
                 use_intensities=False,
                 use_convergence=False,
                 use_potential=False,
                 use_deflections=False, optimizer_class=af.MultiNest,
                 sub_grid_size=2, interp_pixel_scale=None,
                 mask_function=None, cosmology=cosmo.Planck15):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit
        models and hyper passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(GalaxyFitPhase, self).__init__(phase_name=phase_name,
                                             tag_phases=tag_phases,
                                             phase_folders=phase_folders,
                                             optimizer_class=optimizer_class,
                                             cosmology=cosmology)
        self.use_intensities = use_intensities
        self.use_convergence = use_convergence
        self.use_potential = use_potential
        self.use_deflections = use_deflections
        self.galaxies = galaxies
        self.sub_grid_size = sub_grid_size
        self.interp_pixel_scale = interp_pixel_scale
        self.mask_function = mask_function

    def run(self, galaxy_data, results=None, mask=None):
        """
        Run this phase.

        Parameters
        ----------
        galaxy_data
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has
            been executed

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(galaxy_data=galaxy_data, results=results,
                                      mask=mask)

        self.pass_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)

        return self.make_result(result, analysis)

    def make_analysis(self, galaxy_data, results=None, mask=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying
        functions to allow child classes to change the behaviour of the phase.

        Parameters
        ----------
        galaxy_data
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens: Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a
             set of values
        """

        mask = setup_phase_mask(data=galaxy_data[0], mask=mask,
                                mask_function=self.mask_function,
                                inner_mask_radii=None)

        if self.use_intensities or self.use_convergence or self.use_potential:

            galaxy_data = gd.GalaxyFitData(galaxy_data=galaxy_data[0], mask=mask,
                                           sub_grid_size=self.sub_grid_size,
                                           interp_pixel_scale=self.interp_pixel_scale,
                                           use_intensities=self.use_intensities,
                                           use_convergence=self.use_convergence,
                                           use_potential=self.use_potential,
                                           use_deflections_y=self.use_deflections,
                                           use_deflections_x=self.use_deflections)

            return self.AnalysisSingle(galaxy_data=galaxy_data,
                                                 cosmology=self.cosmology,
                                                 results=results)

        elif self.use_deflections:

            galaxy_data_y = gd.GalaxyFitData(galaxy_data=galaxy_data[0], mask=mask,
                                             sub_grid_size=self.sub_grid_size,
                                             interp_pixel_scale=self.interp_pixel_scale,
                                             use_intensities=self.use_intensities,
                                             use_convergence=self.use_convergence,
                                             use_potential=self.use_potential,
                                             use_deflections_y=self.use_deflections,
                                             use_deflections_x=False)

            galaxy_data_x = gd.GalaxyFitData(galaxy_data=galaxy_data[1], mask=mask,
                                             sub_grid_size=self.sub_grid_size,
                                             interp_pixel_scale=self.interp_pixel_scale,
                                             use_intensities=self.use_intensities,
                                             use_convergence=self.use_convergence,
                                             use_potential=self.use_potential,
                                             use_deflections_y=False,
                                             use_deflections_x=self.use_deflections)

            return self.AnalysisDeflections(galaxy_data_y=galaxy_data_y,
                                                      galaxy_data_x=galaxy_data_x,
                                                      cosmology=self.cosmology,
                                                      results=results)

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, cosmology, results):
            super(GalaxyFitPhase.Analysis, self).__init__(cosmology=cosmology,
                                                          results=results)

            self.plot_galaxy_fit_all_at_end_png = \
                af.conf.instance.visualize.get('plots', 'plot_galaxy_fit_all_at_end_png',
                                          bool)
            self.plot_galaxy_fit_all_at_end_fits = \
                af.conf.instance.visualize.get('plots', 'plot_galaxy_fit_all_at_end_fits',
                                          bool)
            self.plot_galaxy_fit_as_subplot = \
                af.conf.instance.visualize.get('plots', 'plot_galaxy_fit_as_subplot', bool)
            self.plot_galaxy_fit_image = \
                af.conf.instance.visualize.get('plots', 'plot_galaxy_fit_image', bool)
            self.plot_galaxy_fit_noise_map = \
                af.conf.instance.visualize.get('plots', 'plot_galaxy_fit_noise_map', bool)
            self.plot_galaxy_fit_model_image = \
                af.conf.instance.visualize.get('plots', 'plot_galaxy_fit_model_image', bool)
            self.plot_galaxy_fit_residual_map = \
                af.conf.instance.visualize.get('plots', 'plot_galaxy_fit_residual_map',
                                          bool)
            self.plot_galaxy_fit_chi_squared_map = \
                af.conf.instance.visualize.get('plots', 'plot_galaxy_fit_chi_squared_map',
                                          bool)

        @classmethod
        def describe(cls, instance):
            return "\nRunning galaxy fit for... \n\nGalaxies::\n{}\n\n".format(
                instance.galaxies)

    # noinspection PyAbstractClass
    class AnalysisSingle(Analysis):

        def __init__(self, galaxy_data, cosmology, results=None):
            super(GalaxyFitPhase.AnalysisSingle, self).__init__(cosmology=cosmology,
                                                                results=results)

            self.galaxy_data = galaxy_data

        def fit(self, instance):
            fit = self.fit_for_instance(instance=instance)
            return fit.figure_of_merit

        def visualize(self, instance, image_path, during_analysis):

            self.plot_count += 1
            fit = self.fit_for_instance(instance=instance)

            if self.plot_galaxy_fit_as_subplot:
                galaxy_fit_plotters.plot_fit_subplot(
                    fit=fit, should_plot_mask=self.should_plot_mask,
                    zoom_around_mask=self.zoom_around_mask,
                    units=self.plot_units,
                    output_path=image_path, output_format='png')

            if during_analysis:

                galaxy_fit_plotters.plot_fit_individuals(
                    fit=fit, should_plot_mask=self.should_plot_mask,
                    zoom_around_mask=self.zoom_around_mask,
                    should_plot_image=self.plot_galaxy_fit_image,
                    should_plot_noise_map=self.plot_galaxy_fit_noise_map,
                    should_plot_model_image=self.plot_galaxy_fit_model_image,
                    should_plot_residual_map=self.plot_galaxy_fit_residual_map,
                    should_plot_chi_squared_map=self.plot_galaxy_fit_chi_squared_map,
                    units=self.plot_units,
                    output_path=image_path, output_format='png')

            elif not during_analysis:

                if self.plot_ray_tracing_all_at_end_png:
                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit, should_plot_mask=self.should_plot_mask,
                        zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=image_path, output_format='png')

                if self.plot_ray_tracing_all_at_end_fits:
                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit, should_plot_mask=self.should_plot_mask,
                        zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path="{}/fits/".format(image_path), output_format='fits')

            return fit

        def fit_for_instance(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the lens_data in
            this lens.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model
                lens_data itself
            """
            return galaxy_fit.GalaxyFit(galaxy_data=self.galaxy_data,
                                        model_galaxies=instance.galaxies)

    # noinspection PyAbstractClass
    class AnalysisDeflections(Analysis):

        def __init__(self, galaxy_data_y, galaxy_data_x, cosmology, results=None):
            super(GalaxyFitPhase.AnalysisDeflections, self).__init__(
                cosmology=cosmology,
                results=results)

            self.galaxy_data_y = galaxy_data_y
            self.galaxy_data_x = galaxy_data_x

        def fit(self, instance):
            fit_y, fit_x = self.fit_for_instance(instance=instance)
            return fit_y.figure_of_merit + fit_x.figure_of_merit

        def visualize(self, instance, image_path, during_analysis):

            output_image_y_path = "{}/fit_y_".format(image_path)
            output_fits_y_path = "{}/fits/fit_y".format(image_path)
            output_image_x_path = "{}/fit_x_".format(image_path)
            output_fits_x_path = "{}/fits/fit_x".format(image_path)

            self.plot_count += 1
            fit_y, fit_x = self.fit_for_instance(instance=instance)

            if self.plot_galaxy_fit_as_subplot:
                galaxy_fit_plotters.plot_fit_subplot(
                    fit=fit_y, should_plot_mask=self.should_plot_mask,
                    zoom_around_mask=self.zoom_around_mask,
                    units=self.plot_units,
                    output_path=output_image_y_path, output_format='png')

                galaxy_fit_plotters.plot_fit_subplot(
                    fit=fit_x, should_plot_mask=self.should_plot_mask,
                    zoom_around_mask=self.zoom_around_mask,
                    units=self.plot_units,
                    output_path=output_image_x_path, output_format='png')

            if during_analysis:

                galaxy_fit_plotters.plot_fit_individuals(
                    fit=fit_y, should_plot_mask=self.should_plot_mask,
                    zoom_around_mask=self.zoom_around_mask,
                    should_plot_image=self.plot_galaxy_fit_image,
                    should_plot_noise_map=self.plot_galaxy_fit_noise_map,
                    should_plot_model_image=self.plot_galaxy_fit_model_image,
                    should_plot_residual_map=self.plot_galaxy_fit_residual_map,
                    should_plot_chi_squared_map=self.plot_galaxy_fit_chi_squared_map,
                    units=self.plot_units,
                    output_path=output_image_y_path, output_format='png')

                galaxy_fit_plotters.plot_fit_individuals(
                    fit=fit_x, should_plot_mask=self.should_plot_mask,
                    zoom_around_mask=self.zoom_around_mask,
                    should_plot_image=self.plot_galaxy_fit_image,
                    should_plot_noise_map=self.plot_galaxy_fit_noise_map,
                    should_plot_model_image=self.plot_galaxy_fit_model_image,
                    should_plot_residual_map=self.plot_galaxy_fit_residual_map,
                    should_plot_chi_squared_map=self.plot_galaxy_fit_chi_squared_map,
                    units=self.plot_units,
                    output_path=output_image_x_path, output_format='png')

            elif not during_analysis:

                if self.plot_ray_tracing_all_at_end_png:
                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit_y, should_plot_mask=self.should_plot_mask,
                        zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=output_image_y_path, output_format='png')

                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit_x, should_plot_mask=self.should_plot_mask,
                        zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=output_image_x_path, output_format='png')

                if self.plot_ray_tracing_all_at_end_fits:
                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit_y, should_plot_mask=self.should_plot_mask,
                        zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=output_fits_y_path, output_format='fits')

                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit_x, should_plot_mask=self.should_plot_mask,
                        zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=output_fits_x_path, output_format='fits')

            return fit_y, fit_x

        def fit_for_instance(self, instance):

            fit_y = galaxy_fit.GalaxyFit(galaxy_data=self.galaxy_data_y,
                                         model_galaxies=instance.galaxies)
            fit_x = galaxy_fit.GalaxyFit(galaxy_data=self.galaxy_data_x,
                                         model_galaxies=instance.galaxies)

            return fit_y, fit_x
