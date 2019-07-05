import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autolens import exc
from autolens.data.array import grids
from autolens.lens import ray_tracing, lens_data as ld, lens_fit, sensitivity_fit
from autolens.lens.plotters import sensitivity_fit_plotters
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import pixelizations as px
from autolens.pipeline import tagging as tag
from autolens.pipeline.phase import phase_extensions
from autolens.pipeline.plotters import phase_plotters
from autolens.pipeline.phase.phase import Phase, setup_phase_mask


class PhaseImaging(Phase):

    def __init__(self,
                 phase_name,
                 tag_phases=True,
                 phase_folders=tuple(),
                 hyper_image_sky=None,
                 hyper_noise_background=None,
                 optimizer_class=af.MultiNest,
                 sub_grid_size=2,
                 bin_up_factor=None,
                 image_psf_shape=None,
                 inversion_psf_shape=None,
                 positions_threshold=None,
                 mask_function=None,
                 inner_mask_radii=None,
                 interp_pixel_scale=None,
                 inversion_pixel_limit=None,
                 cluster_pixel_scale=None,
                 cosmology=cosmo.Planck15,
                 auto_link_priors=False):

        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        inversion_max_pixels : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        cluster_pixel_scale : float or None
            If *True*, the hyper image used to generate the cluster'grids weight map will be binned up to this higher \
            pixel scale to speed up the KMeans clustering algorithm.
        """

        if tag_phases:

            phase_tag = tag.phase_tag_from_phase_settings(
                sub_grid_size=sub_grid_size,
                bin_up_factor=bin_up_factor,
                image_psf_shape=image_psf_shape,
                inversion_psf_shape=inversion_psf_shape,
                positions_threshold=positions_threshold,
                inner_mask_radii=inner_mask_radii,
                interp_pixel_scale=interp_pixel_scale,
                cluster_pixel_scale=cluster_pixel_scale)

        else:

            phase_tag = None

        super(PhaseImaging, self).__init__(
            phase_name=phase_name,
            phase_tag=phase_tag,
            phase_folders=phase_folders,
            tag_phases=tag_phases,
            optimizer_class=optimizer_class,
            cosmology=cosmology,
            auto_link_priors=auto_link_priors)

        self.sub_grid_size = sub_grid_size
        self.bin_up_factor = bin_up_factor
        self.image_psf_shape = image_psf_shape
        self.inversion_psf_shape = inversion_psf_shape
        self.positions_threshold = positions_threshold
        self.mask_function = mask_function
        self.inner_mask_radii = inner_mask_radii
        self.interp_pixel_scale = interp_pixel_scale
        self.inversion_pixel_limit = inversion_pixel_limit
        self.cluster_pixel_scale = cluster_pixel_scale

        inversion_pixel_limit_from_prior = \
            int(af.conf.instance.prior_default.get('pixelizations', 'VoronoiBrightnessImage', 'pixels')[2])

        if self.inversion_pixel_limit is not None:

            self.cluster_pixel_limit = min(inversion_pixel_limit_from_prior, self.inversion_pixel_limit)

        else:

            self.cluster_pixel_limit = inversion_pixel_limit_from_prior

        self.hyper_image_sky = hyper_image_sky
        self.hyper_noise_background = hyper_noise_background

    @property
    def uses_hyper_images(self) -> bool:
        return False

    @property
    def uses_inversion(self) -> bool:
        return False

    @property
    def uses_cluster_inversion(self) -> bool:
        return False

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def modify_image(self, image, results):
        """
        Customize an lens_data. e.g. removing lens light.

        Parameters
        ----------
        image: scaled_array.ScaledSquarePixelArray
            An lens_data that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous lens

        Returns
        -------
        lens_data: scaled_array.ScaledSquarePixelArray
            The modified image (not changed by default)
        """
        return image

    def run(self, data, results=None, mask=None, positions=None):
        """
        Run this phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        data: scaled_array.ScaledSquarePixelArray
            An lens_data that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(data=data, results=results, mask=mask,
                                      positions=positions)

        self.pass_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)

        return self.make_result(result, analysis)

    def make_analysis(self, data, results=None, mask=None, positions=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        data: im.CCD
            An lens_data that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """

        mask = setup_phase_mask(data=data, mask=mask, mask_function=self.mask_function,
                                inner_mask_radii=self.inner_mask_radii)

        if self.positions_threshold is not None and positions is not None:
            positions = list(map(lambda position_set: np.asarray(position_set), positions))
        elif self.positions_threshold is None:
            positions = None
        elif self.positions_threshold is not None and positions is None:
            raise exc.PhaseException(
                'You have specified for a phase to use positions, but not input positions to the '
                'pipeline when you ran it.')

        lens_data = ld.LensData(
            ccd_data=data,
            mask=mask,
            sub_grid_size=self.sub_grid_size,
            image_psf_shape=self.image_psf_shape,
            positions=positions,
            interp_pixel_scale=self.interp_pixel_scale,
            cluster_pixel_scale=self.cluster_pixel_scale,
            cluster_pixel_limit=self.cluster_pixel_limit,
            uses_inversion=self.uses_inversion,
            uses_cluster_inversion=self.uses_cluster_inversion)

        modified_image = self.modify_image(
            image=lens_data.unmasked_image, results=results)

        lens_data = lens_data.new_lens_data_with_modified_image(
            modified_image=modified_image)

        if self.bin_up_factor is not None:

            lens_data = lens_data.new_lens_data_with_binned_up_ccd_data_and_mask(
                bin_up_factor=self.bin_up_factor)

        self.output_phase_info()

        analysis = self.Analysis(
            lens_data=lens_data,
            cosmology=self.cosmology,
            positions_threshold=self.positions_threshold,
            image_path=self.optimizer.image_path,
            results=results,
            inversion_pixel_limit=self.inversion_pixel_limit)

        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.optimizer.phase_output_path, 'phase.info')

        with open(file_phase_info, 'w') as phase_info:
            phase_info.write('Optimizer = {} \n'.format(type(self.optimizer).__name__))
            phase_info.write('Sub-grid size = {} \n'.format(self.sub_grid_size))
            phase_info.write('Image PSF shape = {} \n'.format(self.image_psf_shape))
            phase_info.write(
                'Pixelization PSF shape = {} \n'.format(self.inversion_psf_shape))
            phase_info.write(
                'Positions Threshold = {} \n'.format(self.positions_threshold))
            phase_info.write('Cosmology = {} \n'.format(self.cosmology))
            phase_info.write('Auto Link Priors = {} \n'.format(self.auto_link_priors))

            phase_info.close()

    def extend_with_inversion_phase(self):
        return phase_extensions.CombinedHyperPhase(phase=self, hyper_phase_classes=(phase_extensions.InversionPhase,))

    def extend_with_hyper_and_inversion_phases(
            self, hyper_galaxy=False, inversion=False,
            include_background_sky=False, include_background_noise=False):

        if hyper_galaxy:
            if not include_background_noise:
                phase_hyper_galaxy = phase_extensions.HyperGalaxyPhase
            else:
                phase_hyper_galaxy = phase_extensions.HyperGalaxyBackgroundPhase
        else:
            phase_hyper_galaxy = None

        if inversion:
            if not include_background_sky and not include_background_noise:
                phase_inversion = phase_extensions.InversionPhase
            elif include_background_sky and not include_background_noise:
                phase_inversion = phase_extensions.InversionBackgroundSkyPhase
            elif not include_background_sky and include_background_noise:
                phase_inversion = phase_extensions.InversionBackgroundNoisePhase
            else:
                phase_inversion = phase_extensions.InversionBackgroundBothPhase
        else:
            phase_inversion = None

        hyper_phase_classes = filter(None, (phase_hyper_galaxy, phase_inversion))

        return phase_extensions.CombinedHyperPhase(
            phase=self, hyper_phase_classes=hyper_phase_classes)

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, lens_data, cosmology, positions_threshold, inversion_pixel_limit=None, image_path=None,
                     results=None):

            super(PhaseImaging.Analysis, self).__init__(cosmology=cosmology,
                                                        results=results)

            self.lens_data = lens_data

            self.positions_threshold = positions_threshold

            self.should_plot_image_plane_pix = af.conf.instance.visualize.get(
                'figures', 'plot_image_plane_adaptive_pixelization_grid', bool)

            self.plot_data_as_subplot = \
                af.conf.instance.visualize.get('plots', 'plot_data_as_subplot', bool)

            self.plot_data_image = \
                af.conf.instance.visualize.get('plots', 'plot_data_image', bool)

            self.plot_data_noise_map = \
                af.conf.instance.visualize.get('plots', 'plot_data_noise_map', bool)

            self.plot_data_psf = \
                af.conf.instance.visualize.get('plots', 'plot_data_psf', bool)

            self.plot_data_signal_to_noise_map = \
                af.conf.instance.visualize.get('plots', 'plot_data_signal_to_noise_map', bool)

            self.plot_data_absolute_signal_to_noise_map = \
                af.conf.instance.visualize.get('plots', 'plot_data_absolute_signal_to_noise_map',bool)

            self.plot_data_potential_chi_squared_map = \
                af.conf.instance.visualize.get('plots', 'plot_data_potential_chi_squared_map', bool)


            self.plot_lens_fit_all_at_end_png = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_all_at_end_png',
                                             bool)
            self.plot_lens_fit_all_at_end_fits = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_all_at_end_fits', bool)

            self.plot_lens_fit_as_subplot = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_as_subplot', bool)

            self.plot_lens_fit_of_planes_as_subplot = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_of_planes_as_subplot', bool)

            self.plot_lens_fit_inversion_as_subplot = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_inversion_as_subplot', bool)

            self.plot_lens_fit_image = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_image', bool)

            self.plot_lens_fit_noise_map = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_noise_map', bool)

            self.plot_lens_fit_signal_to_noise_map = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_signal_to_noise_map', bool)

            self.plot_lens_fit_model_image = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_model_image', bool)

            self.plot_lens_fit_residual_map = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_residual_map', bool)

            self.plot_lens_fit_normalized_residual_map = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_normalized_residual_map', bool)

            self.plot_lens_fit_chi_squared_map = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_chi_squared_map', bool)

            self.plot_lens_fit_contribution_maps = \
                af.conf.instance.visualize.get('plots','plot_lens_fit_contribution_maps', bool)

            self.plot_lens_fit_pixelization_residual_map = \
                af.conf.instance.visualize.get('plots','plot_lens_fit_pixelization_residual_map', bool)

            self.plot_lens_fit_pixelization_normalized_residuals = \
                af.conf.instance.visualize.get('plots','plot_lens_fit_pixelization_normalized_residual_map', bool)

            self.plot_lens_fit_pixelization_chi_squared_map = \
                af.conf.instance.visualize.get('plots','plot_lens_fit_pixelization_chi_squared_map', bool)

            self.plot_lens_fit_pixelization_regularization_weights = \
                af.conf.instance.visualize.get('plots','plot_lens_fit_pixelization_regularization_weight_map', bool)

            self.plot_lens_fit_subtracted_images_of_planes = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_subtracted_images_of_planes', bool)

            self.plot_lens_fit_model_images_of_planes = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_model_images_of_planes', bool)

            self.plot_lens_fit_plane_images_of_planes = \
                af.conf.instance.visualize.get('plots', 'plot_lens_fit_plane_images_of_planes', bool)

            self.inversion_pixel_limit = inversion_pixel_limit

            mask = self.lens_data.mask_2d if self.should_plot_mask else None
            positions = self.lens_data.positions if self.should_plot_positions else None

            phase_plotters.plot_ccd_for_phase(
                ccd_data=self.lens_data.ccd_data, mask=mask, positions=positions,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
                units=self.plot_units,
                should_plot_as_subplot=self.plot_data_as_subplot,
                should_plot_image=self.plot_data_image,
                should_plot_noise_map=self.plot_data_noise_map,
                should_plot_psf=self.plot_data_psf,
                should_plot_signal_to_noise_map=self.plot_data_signal_to_noise_map,
                should_plot_absolute_signal_to_noise_map=self.plot_data_absolute_signal_to_noise_map,
                should_plot_potential_chi_squared_map=self.plot_data_potential_chi_squared_map,
                visualize_path=image_path)

            self.plot_hyper_model_image = \
                af.conf.instance.visualize.get('plots', 'plot_hyper_model_image', bool)

            self.plot_hyper_galaxy_images = \
                af.conf.instance.visualize.get('plots', 'plot_hyper_galaxy_images', bool)

            self.plot_hyper_galaxy_cluster_images = \
                af.conf.instance.visualize.get('plots', 'plot_hyper_galaxy_cluster_images', bool)

            self.preload_pixelization_grid = None

            if self.last_results is not None:

                self.hyper_galaxy_image_1d_path_dict = self.last_results.hyper_galaxy_image_1d_path_dict_from_mask(
                    mask=lens_data.mask_2d)

                self.hyper_model_image_1d = self.last_results.hyper_model_image_1d_from_mask(mask=lens_data.mask_2d)

                self.hyper_galaxy_cluster_image_1d_path_dict = \
                    self.last_results.hyper_galaxy_cluster_image_1d_path_dict_from_cluster(cluster=lens_data.cluster)

                phase_plotters.plot_hyper_images_for_phase(
                    hyper_model_image_2d=mask.scaled_array_2d_from_array_1d(
                        array_1d=self.hyper_model_image_1d),
                    hyper_galaxy_image_2d_path_dict=
                    self.last_results.hyper_galaxy_image_2d_path_dict_from_mask(mask=mask),
                    hyper_galaxy_cluster_image_2d_path_dict=
                    self.last_results.hyper_galaxy_cluster_image_2d_path_dict_from_cluster(cluster=lens_data.cluster),
                    mask=lens_data.mask_2d,
                    cluster=lens_data.cluster,
                    extract_array_from_mask=self.extract_array_from_mask,
                    zoom_around_mask=self.zoom_around_mask,
                    units=self.plot_units,
                    should_plot_hyper_model_image=self.plot_hyper_model_image,
                    should_plot_hyper_galaxy_images=self.plot_hyper_galaxy_images,
                    should_plot_hyper_galaxy_cluster_images=self.plot_hyper_galaxy_cluster_images,
                    visualize_path=image_path)

                if hasattr(self.results.last, 'inversion'):

                    self.preload_pixelization_grid = self.results.last.inversion.most_likely_image_plane_pixelization_grid

        def fit(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the lens_data in this lens.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit : Fit
                A fractional value indicating how well this model fit and the model lens_data itself
            """
            self.check_positions_trace_within_threshold(instance=instance)
            self.check_inversion_pixels_are_below_limit(instance=instance)
            tracer = self.tracer_for_instance(instance=instance)

            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_noise_background = self.hyper_noise_background_for_instance(instance=instance)

            fit = self.fit_for_tracers(
                tracer=tracer,
                padded_tracer=None,
                hyper_image_sky=hyper_image_sky,
                hyper_noise_background=hyper_noise_background)

            return fit.figure_of_merit

        def check_for_previously_masked_values(self, array):
            if not np.all(array) != 0.0 and not np.all(array == 0):
                raise exc.PhaseException(
                    'When mapping a 2D array to a 1D array using lens data, a value '
                    'encountered was 0.0 and therefore masked in a previous phase.')

        def associate_images(self, instance: af.ModelInstance) -> af.ModelInstance:
            """
            Takes images from the last result, if there is one, and associates them with galaxies in this phase where
            full-path galaxy names match.

            If the galaxy collection has a different name then an association is not made.

            e.g.
            lens_galaxies.lens will match with:
                lens_galaxies.lens
            but not with:
                galaxies.lens
                lens_galaxies.source

            Parameters
            ----------
            instance
                A model instance with 0 or more galaxies in its tree

            Returns
            -------
            instance
               The input instance with images associated with galaxies where possible.
            """
            if hasattr(self, 'hyper_galaxy_image_1d_path_dict'):
                for galaxy_path, galaxy in instance.path_instance_tuples_for_class(g.Galaxy):
                    if galaxy_path in self.hyper_galaxy_image_1d_path_dict:
                        galaxy.hyper_model_image_1d = self.hyper_model_image_1d
                        galaxy.hyper_galaxy_image_1d = self.hyper_galaxy_image_1d_path_dict[galaxy_path]
                        if self.hyper_galaxy_cluster_image_1d_path_dict is not None:
                            galaxy.hyper_galaxy_cluster_image_1d = self.hyper_galaxy_cluster_image_1d_path_dict[galaxy_path]
            return instance

        def add_grids_to_grid_stack(self, galaxies, grid_stack):

            if self.preload_pixelization_grid is None:

                for galaxy in galaxies:
                    if galaxy.pixelization is not None:
                        if galaxy.pixelization.uses_pixelization_grid:

                            if isinstance(galaxy.pixelization, px.VoronoiMagnification):

                                sparse_to_regular_grid = grids.SparseToRegularGrid.from_unmasked_2d_grid_shape_and_regular_grid(
                                    unmasked_sparse_shape=galaxy.pixelization.shape, regular_grid=grid_stack.regular)

                            elif isinstance(galaxy.pixelization, px.VoronoiBrightnessImage):

                                cluster_weight_map = galaxy.pixelization.cluster_weight_map_from_hyper_image(
                                        hyper_image=galaxy.hyper_galaxy_cluster_image_1d)

                                sparse_to_regular_grid = \
                                    grids.SparseToRegularGrid.from_total_pixels_cluster_grid_and_cluster_weight_map(
                                        total_pixels=galaxy.pixelization.pixels, cluster_grid=self.lens_data.cluster,
                                        regular_grid=self.lens_data.grid_stack.regular, cluster_weight_map=cluster_weight_map,
                                        seed=1)

                            else:

                                raise exc.PhaseException('The pixelization of a galaxy uses a pixelization grid, but was not a viable'
                                                         'type in the grid stack calculation method')

                            pixelization_grid = grids.PixelizationGrid(
                                arr=sparse_to_regular_grid.sparse, regular_to_pixelization=sparse_to_regular_grid.regular_to_sparse)

                            return grid_stack.new_grid_stack_with_grids_added(pixelization=pixelization_grid)

            else:

                return grid_stack.new_grid_stack_with_grids_added(pixelization=self.preload_pixelization_grid)

            return grid_stack

        def visualize(self, instance, image_path, during_analysis):

            instance = self.associate_images(instance=instance)

            mask = self.lens_data.mask_2d if self.should_plot_mask else None
            positions = self.lens_data.positions if self.should_plot_positions else None

            tracer = self.tracer_for_instance(instance)

            phase_plotters.plot_ray_tracing_for_phase(
                tracer=tracer, during_analysis=during_analysis, mask=mask,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask, positions=positions,
                units=self.plot_units,
                should_plot_as_subplot=self.plot_ray_tracing_as_subplot,
                should_plot_all_at_end_png=self.plot_ray_tracing_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_ray_tracing_all_at_end_fits,
                should_plot_image_plane_image=self.plot_ray_tracing_image_plane_image,
                should_plot_source_plane=self.plot_ray_tracing_source_plane,
                should_plot_convergence=self.plot_ray_tracing_convergence,
                should_plot_potential=self.plot_ray_tracing_potential,
                should_plot_deflections=self.plot_ray_tracing_deflections,
                visualize_path=image_path)

            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_noise_background = self.hyper_noise_background_for_instance(instance=instance)

            padded_tracer = self.padded_tracer_for_instance(instance=instance)
            fit = self.fit_for_tracers(
                tracer=tracer, padded_tracer=padded_tracer,
                hyper_image_sky=hyper_image_sky, hyper_noise_background=hyper_noise_background)

            phase_plotters.plot_lens_fit_for_phase(
                fit=fit, during_analysis=during_analysis,
                should_plot_mask=self.should_plot_mask,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
                positions=positions,
                should_plot_image_plane_pix=self.should_plot_image_plane_pix,
                should_plot_all_at_end_png=self.plot_lens_fit_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_lens_fit_all_at_end_fits,
                should_plot_fit_as_subplot=self.plot_lens_fit_as_subplot,
                should_plot_fit_of_planes_as_subplot=self.plot_lens_fit_of_planes_as_subplot,
                should_plot_inversion_as_subplot=self.plot_lens_fit_inversion_as_subplot,
                should_plot_image=self.plot_lens_fit_image,
                should_plot_noise_map=self.plot_lens_fit_noise_map,
                should_plot_signal_to_noise_map=self.plot_lens_fit_signal_to_noise_map,
                should_plot_model_image=self.plot_lens_fit_model_image,
                should_plot_residual_map=self.plot_lens_fit_residual_map,
                should_plot_normalized_residual_map=self.plot_lens_fit_normalized_residual_map,
                should_plot_chi_squared_map=self.plot_lens_fit_chi_squared_map,
                should_plot_pixelization_residual_map=self.plot_lens_fit_pixelization_residual_map,
                should_plot_pixelization_normalized_residual_map=self.plot_lens_fit_normalized_residual_map,
                should_plot_pixelization_chi_squared_map=self.plot_lens_fit_pixelization_chi_squared_map,
                should_plot_pixelization_regularization_weights=self.plot_lens_fit_pixelization_regularization_weights,
                should_plot_subtracted_images_of_planes=self.plot_lens_fit_subtracted_images_of_planes,
                should_plot_model_images_of_planes=self.plot_lens_fit_model_images_of_planes,
                should_plot_plane_images_of_planes=self.plot_lens_fit_plane_images_of_planes,
                units=self.plot_units,
                visualize_path=image_path)

        def hyper_image_sky_for_instance(self, instance):

            if hasattr(instance, 'hyper_image_sky'):
                return instance.hyper_image_sky
            else:
                return None

        def hyper_noise_background_for_instance(self, instance):

            if hasattr(instance, 'hyper_noise_background'):
                return instance.hyper_noise_background
            else:
                return None

        def fit_for_tracers(self, tracer, padded_tracer, hyper_image_sky, hyper_noise_background):

            return lens_fit.LensDataFit.for_data_and_tracer(
                lens_data=self.lens_data,
                tracer=tracer,
                padded_tracer=padded_tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_noise_background=hyper_noise_background)

        def check_positions_trace_within_threshold(self, instance):

            if self.lens_data.positions is not None:

                tracer = ray_tracing.TracerImageSourcePlanesPositions(
                    lens_galaxies=instance.lens_galaxies,
                    image_plane_positions=self.lens_data.positions)
                fit = lens_fit.LensPositionFit(positions=tracer.source_plane.positions,
                                               noise_map=self.lens_data.pixel_scale)

                if not fit.maximum_separation_within_threshold(
                        self.positions_threshold):
                    raise exc.RayTracingException

        def check_inversion_pixels_are_below_limit(self, instance):
            raise NotImplementedError

        def map_to_1d(self, data):
            """Convenience method"""
            return self.lens_data.mask.array_1d_from_array_2d(data)


class MultiPlanePhase(PhaseImaging):
    """
    Fit a simple source and lens system.
    """

    hyper_image_sky = af.PhaseProperty("hyper_image_sky")
    hyper_noise_background = af.PhaseProperty("hyper_noise_background")
    galaxies = af.PhaseProperty("galaxies")

    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), galaxies=None,
                 hyper_image_sky=None, hyper_noise_background=None,
                 optimizer_class=af.MultiNest,
                 sub_grid_size=2, bin_up_factor=None, image_psf_shape=None,
                 positions_threshold=None,
                 mask_function=None,
                 inner_mask_radii=None,
                 inversion_pixel_limit=None,
                 cluster_pixel_scale=None,
                 cosmology=cosmo.Planck15,
                 auto_link_priors=False):
        """
        A phase with a simple source/lens model

        Parameters
        ----------
        galaxies : [g.Galaxy] | [gm.GalaxyModel]
            A galaxy that acts as a gravitational lens or is being lensed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        inversion_max_pixels : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        cluster_pixel_scale : float or None
            If *True*, the hyper image used to generate the cluster'grids weight map will be binned up to this higher \
            pixel scale to speed up the KMeans clustering algorithm.
        """

        super(MultiPlanePhase, self).__init__(
            phase_name=phase_name,
            tag_phases=tag_phases,
            phase_folders=phase_folders,
            hyper_image_sky=hyper_image_sky,
            hyper_noise_background=hyper_noise_background,
            optimizer_class=optimizer_class,
            sub_grid_size=sub_grid_size,
            bin_up_factor=bin_up_factor,
            image_psf_shape=image_psf_shape,
            positions_threshold=positions_threshold,
            mask_function=mask_function,
            inner_mask_radii=inner_mask_radii,
            inversion_pixel_limit=inversion_pixel_limit,
            cluster_pixel_scale=cluster_pixel_scale,
            cosmology=cosmology,
            auto_link_priors=auto_link_priors)

        self.galaxies = galaxies

    @property
    def uses_hyper_images(self):
        if self.galaxies:
            return any([galaxy.uses_hyper_images for galaxy in self.galaxies])
        else:
            return False

    @property
    def uses_inversion(self):
        if self.galaxies:
            for galaxy in self.galaxies:
                if galaxy.uses_inversion:
                    return True
        return False

    @property
    def uses_cluster_inversion(self):
        if self.galaxies:
            for galaxy in self.galaxies:
                if galaxy.uses_cluster_inversion:
                    return True
        return False

    class Analysis(PhaseImaging.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

        def tracer_for_instance(self, instance):

            instance = self.associate_images(instance=instance)

            image_plane_grid_stack = self.add_grids_to_grid_stack(
                galaxies=instance.galaxies, grid_stack=self.lens_data.grid_stack)

            return ray_tracing.TracerMultiPlanes(galaxies=instance.galaxies,
                                                 image_plane_grid_stack=image_plane_grid_stack,
                                                 border=self.lens_data.border,
                                                 cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):

            instance = self.associate_images(instance=instance)

            return ray_tracing.TracerMultiPlanes(galaxies=instance.galaxies,
                                                 image_plane_grid_stack=self.lens_data.padded_grid_stack,
                                                 cosmology=self.cosmology)

        def check_inversion_pixels_are_below_limit(self, instance):

            if self.inversion_pixel_limit is not None:
                if instance.galaxies:
                    for galaxy in instance.galaxies:
                        if galaxy.has_pixelization:
                            if galaxy.pixelization.pixels > self.inversion_pixel_limit:
                                raise exc.PixelizationException

        @classmethod
        def describe(cls, instance):
            return "\nRunning multi-plane for... \n\nGalaxies:\n{}\n\n".format(
                instance.galaxies)


class LensSourcePlanePhase(PhaseImaging):
    """
    Fit a simple source and lens system.
    """

    hyper_image_sky = af.PhaseProperty("hyper_image_sky")
    hyper_noise_background = af.PhaseProperty("hyper_noise_background")
    lens_galaxies = af.PhaseProperty("lens_galaxies")
    source_galaxies = af.PhaseProperty("source_galaxies")

    def __init__(self,
                 phase_name,
                 tag_phases=True,
                 phase_folders=tuple(),
                 lens_galaxies=None,
                 source_galaxies=None,
                 hyper_image_sky=None,
                 hyper_noise_background=None,
                 optimizer_class=af.MultiNest,
                 sub_grid_size=2, bin_up_factor=None,
                 image_psf_shape=None,
                 positions_threshold=None,
                 mask_function=None,
                 inner_mask_radii=None,
                 interp_pixel_scale=None,
                 inversion_pixel_limit=None,
                 cluster_pixel_scale=None,
                 cosmology=cosmo.Planck15,
                 auto_link_priors=False):
        """
        A phase with a simple source/lens model

        Parameters
        ----------
        lens_galaxies : [g.Galaxy] | [gm.GalaxyModel]
            A galaxy that acts as a gravitational lens
        source_galaxies: [g.Galaxy] | [gm.GalaxyModel]
            A galaxy that is being lensed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        inversion_max_pixels : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        cluster_pixel_scale : float or None
            If *True*, the hyper image used to generate the cluster'grids weight map will be binned up to this higher \
            pixel scale to speed up the KMeans clustering algorithm.
        """
        super(LensSourcePlanePhase, self).__init__(
            phase_name=phase_name,
            tag_phases=tag_phases,
            phase_folders=phase_folders,
            hyper_image_sky=hyper_image_sky,
            hyper_noise_background=hyper_noise_background,
            optimizer_class=optimizer_class,
            sub_grid_size=sub_grid_size,
            bin_up_factor=bin_up_factor,
            image_psf_shape=image_psf_shape,
            positions_threshold=positions_threshold,
            mask_function=mask_function,
            interp_pixel_scale=interp_pixel_scale,
            inner_mask_radii=inner_mask_radii,
            inversion_pixel_limit=inversion_pixel_limit,
            cluster_pixel_scale=cluster_pixel_scale,
            cosmology=cosmology,
            auto_link_priors=auto_link_priors)

        self.lens_galaxies = lens_galaxies or []
        self.source_galaxies = source_galaxies or []

    @property
    def uses_inversion(self):

        if self.lens_galaxies:
            for galaxy_model in self.lens_galaxies:
                if galaxy_model.uses_inversion:
                    return True

        if self.source_galaxies:
            for galaxy_model in self.source_galaxies:
                if galaxy_model.uses_inversion:
                    return True
        return False

    @property
    def uses_cluster_inversion(self):

        if self.lens_galaxies:
            for galaxy in self.lens_galaxies:
                if galaxy.uses_cluster_inversion:
                    return True

        if self.source_galaxies:
            for galaxy in self.source_galaxies:
                if galaxy.uses_cluster_inversion:
                    return True
        return False

    @property
    def uses_hyper_images(self):
        return any([galaxy.uses_hyper_images for galaxy in self.lens_galaxies + self.source_galaxies])

    class Analysis(PhaseImaging.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

        def tracer_for_instance(self, instance):

            instance = self.associate_images(instance=instance)

            image_plane_grid_stack = self.add_grids_to_grid_stack(
                galaxies=instance.source_galaxies, grid_stack=self.lens_data.grid_stack)

            return ray_tracing.TracerImageSourcePlanes(
                lens_galaxies=instance.lens_galaxies,
                source_galaxies=instance.source_galaxies,
                image_plane_grid_stack=image_plane_grid_stack,
                border=self.lens_data.border, cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):

            instance = self.associate_images(instance=instance)

            return ray_tracing.TracerImageSourcePlanes(
                lens_galaxies=instance.lens_galaxies,
                source_galaxies=instance.source_galaxies,
                image_plane_grid_stack=self.lens_data.padded_grid_stack,
                cosmology=self.cosmology)

        def check_inversion_pixels_are_below_limit(self, instance):

            if self.inversion_pixel_limit is not None:

                if instance.lens_galaxies:
                    for galaxy in instance.lens_galaxies:
                        if galaxy.has_pixelization:
                            if galaxy.pixelization.pixels > self.inversion_pixel_limit:
                                raise exc.PixelizationException

                if instance.source_galaxies:
                    for galaxy in instance.source_galaxies:
                        if galaxy.has_pixelization:
                            if galaxy.pixelization.pixels > self.inversion_pixel_limit:
                                raise exc.PixelizationException

        @classmethod
        def describe(cls, instance):
            return "\nRunning lens/source lens for... \n\nLens Galaxy:\n{}\n\nSource " \
                   "Galaxy:\n{}\n\n".format(
                instance.lens_galaxies, instance.source_galaxies)

    class Result(PhaseImaging.Result):
        @property
        def unmasked_lens_plane_model_image(self):
            return self.most_likely_fit.unmasked_blurred_image_plane_image_of_planes[0]

        @property
        def unmasked_source_plane_model_image(self):
            return self.most_likely_fit.unmasked_blurred_image_plane_image_of_planes[1]


class LensPlanePhase(PhaseImaging):
    """
    Fit only the lens galaxy light.
    """

    hyper_image_sky = af.PhaseProperty("hyper_image_sky")
    hyper_noise_background = af.PhaseProperty("hyper_noise_background")
    lens_galaxies = af.PhaseProperty("lens_galaxies")

    def __init__(self,
                 phase_name,
                 tag_phases=True,
                 phase_folders=tuple(),
                 lens_galaxies=None,
                 hyper_image_sky=None,
                 hyper_noise_background=None,
                 optimizer_class=af.MultiNest,
                 sub_grid_size=2,
                 bin_up_factor=None,
                 image_psf_shape=None,
                 mask_function=None,
                 inner_mask_radii=None,
                 cosmology=cosmo.Planck15,
                 auto_link_priors=False):

        super(LensPlanePhase, self).__init__(
            phase_name=phase_name,
            tag_phases=tag_phases,
            phase_folders=phase_folders,
            hyper_image_sky=hyper_image_sky,
            hyper_noise_background=hyper_noise_background,
            optimizer_class=optimizer_class,
            sub_grid_size=sub_grid_size,
            bin_up_factor=bin_up_factor,
            image_psf_shape=image_psf_shape,
            mask_function=mask_function,
            inner_mask_radii=inner_mask_radii,
            cosmology=cosmology,
            auto_link_priors=auto_link_priors)

        self.lens_galaxies = lens_galaxies

    @property
    def uses_hyper_images(self):
        return any([galaxy.uses_hyper_images for galaxy in self.lens_galaxies])

    @property
    def uses_inversion(self):
        if self.lens_galaxies:
            for galaxy_model in self.lens_galaxies:
                if galaxy_model.pixelization is not None:
                    return True
        return False

    class Analysis(PhaseImaging.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

        def tracer_for_instance(self, instance):
            instance = self.associate_images(instance=instance)
            return ray_tracing.TracerImagePlane(lens_galaxies=instance.lens_galaxies,
                                                image_plane_grid_stack=self.lens_data.grid_stack,
                                                cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):
            instance = self.associate_images(instance=instance)
            return ray_tracing.TracerImagePlane(lens_galaxies=instance.lens_galaxies,
                                                image_plane_grid_stack=self.lens_data.padded_grid_stack,
                                                cosmology=self.cosmology)

        def check_inversion_pixels_are_below_limit(self, instance):

            pass

        @classmethod
        def describe(cls, instance):
            return "\nRunning lens lens for... \n\nLens Galaxy::\n{}\n\n".format(
                instance.lens_galaxies)

    class Result(PhaseImaging.Result):

        @property
        def unmasked_lens_plane_model_image(self):
            return self.most_likely_fit.unmasked_blurred_image_plane_image_of_planes[0]


class SensitivityPhase(PhaseImaging):

    lens_galaxies = af.PhaseProperty("lens_galaxies")
    source_galaxies = af.PhaseProperty("source_galaxies")
    sensitive_galaxies = af.PhaseProperty("sensitive_galaxies")

    def __init__(self, phase_name, tag_phases=None, phase_folders=tuple(),
                 lens_galaxies=None, source_galaxies=None,
                 sensitive_galaxies=None,
                 optimizer_class=af.MultiNest, sub_grid_size=2,
                 bin_up_factor=None, mask_function=None,
                 cosmology=cosmo.Planck15):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(SensitivityPhase, self).__init__(phase_name=phase_name,
                                               tag_phases=tag_phases,
                                               phase_folders=phase_folders,
                                               optimizer_class=optimizer_class,
                                               sub_grid_size=sub_grid_size,
                                               bin_up_factor=bin_up_factor,
                                               mask_function=mask_function,
                                               cosmology=cosmology)

        self.lens_galaxies = lens_galaxies or []
        self.source_galaxies = source_galaxies or []
        self.sensitive_galaxies = sensitive_galaxies or []

    # noinspection PyAbstractClass
    class Analysis(PhaseImaging.Analysis):

        def __init__(self, lens_data, cosmology, phase_name, results=None):
            self.lens_data = lens_data
            super(PhaseImaging.Analysis, self).__init__(cosmology=cosmology,
                                                        results=results)

        def fit(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the lens_data in this lens.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model lens_data itself
            """
            tracer_normal = self.tracer_normal_for_instance(instance)
            tracer_sensitive = self.tracer_sensitive_for_instance(instance)
            fit = self.fit_for_tracers(tracer_normal=tracer_normal,
                                       tracer_sensitive=tracer_sensitive)
            return fit.figure_of_merit

        def visualize(self, instance, image_path, during_analysis):
            self.plot_count += 1

            tracer_normal = self.tracer_normal_for_instance(instance)
            tracer_sensitive = self.tracer_sensitive_for_instance(instance)

            fit = self.fit_for_tracers(
                tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)

            phase_plotters.plot_ray_tracing_for_phase(
                tracer=tracer_normal, during_analysis=during_analysis, mask=None, positions=None,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
                units=self.plot_units,
                should_plot_as_subplot=self.plot_ray_tracing_as_subplot,
                should_plot_all_at_end_png=self.plot_ray_tracing_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_ray_tracing_all_at_end_fits,
                should_plot_image_plane_image=self.plot_ray_tracing_image_plane_image,
                should_plot_source_plane=self.plot_ray_tracing_source_plane,
                should_plot_convergence=self.plot_ray_tracing_convergence,
                should_plot_potential=self.plot_ray_tracing_potential,
                should_plot_deflections=self.plot_ray_tracing_deflections,
                visualize_path=image_path)

            phase_plotters.plot_ray_tracing_for_phase(
                tracer=tracer_sensitive, during_analysis=during_analysis, mask=None, positions=None,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
                units=self.plot_units,
                should_plot_as_subplot=self.plot_ray_tracing_as_subplot,
                should_plot_all_at_end_png=self.plot_ray_tracing_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_ray_tracing_all_at_end_fits,
                should_plot_image_plane_image=self.plot_ray_tracing_image_plane_image,
                should_plot_source_plane=self.plot_ray_tracing_source_plane,
                should_plot_convergence=self.plot_ray_tracing_convergence,
                should_plot_potential=self.plot_ray_tracing_potential,
                should_plot_deflections=self.plot_ray_tracing_deflections,
                visualize_path=image_path)

            sensitivity_fit_plotters.plot_fit_subplot(fit=fit, output_path=image_path,
                                                      output_format='png')

            return fit

        def tracer_normal_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(
                lens_galaxies=instance.lens_galaxies,
                source_galaxies=instance.source_galaxies,
                image_plane_grid_stack=self.lens_data.grid_stack,
                border=self.lens_data.border)

        def tracer_sensitive_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(
                lens_galaxies=instance.lens_galaxies + instance.sensitive_galaxies,
                source_galaxies=instance.source_galaxies,
                image_plane_grid_stack=self.lens_data.grid_stack,
                border=self.lens_data.border)

        def fit_for_tracers(self, tracer_normal, tracer_sensitive):
            return sensitivity_fit.fit_lens_data_with_sensitivity_tracers(
                lens_data=self.lens_data,
                tracer_normal=tracer_normal,
                tracer_sensitive=tracer_sensitive)

        @classmethod
        def describe(cls, instance):
            return "\nRunning lens/source lens for... \n\nLens Galaxy:\n{}\n\nSource " \
                   "Galaxy:\n{}\n\n Sensitive " \
                   "Galaxy\n{}\n\n ".format(instance.lens_galaxies,
                                            instance.source_galaxies,
                                            instance.sensitive_galaxies)
