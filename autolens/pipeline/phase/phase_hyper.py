import copy

import numpy as np

import autofit as af
from autolens import exc
from autolens.lens import lens_data as ld, lens_fit
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import pixelizations as px
from autolens.model.inversion import regularization as rg
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline.phase.phase import setup_phase_mask


class HyperPixelizationPhase(phase_imaging.PhaseImaging, af.HyperPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def run(self, data, results=None, mask=None, positions=None):
        """
        Run the phase, overriding the optimizer's variable instance with one created to
        only fit pixelization hyperparameters.
        """
        variable = copy.deepcopy(results.last.variable)
        HyperPixelizationPhase.transfer_classes(results.last.constant, variable)
        self.optimizer.variable = variable
        new_result = super().run(data, results=results, mask=mask, positions=positions)
        result = results.last
        result.hyper = new_result
        return result

    @staticmethod
    def transfer_classes(instance, mapper):
        """
        Recursively overwrite priors in the mapper with constant values from the
        instance except where the containing class is associated with pixelization.
        Parameters
        ----------
        instance
            The best fit from the previous phase
        mapper
            The prior variable from the previous phase
        """
        for key, instance_value in instance.__dict__.items():
            try:
                mapper_value = getattr(mapper, key)
                if isinstance(mapper_value, af.Prior):
                    setattr(mapper, key, instance_value)
                if not (isinstance(instance_value, px.Pixelization) or isinstance(
                        instance_value, rg.Regularization)):
                    try:
                        HyperPixelizationPhase.transfer_classes(instance_value,
                                                                mapper_value)
                    except AttributeError:
                        setattr(mapper, key, instance_value)
            except AttributeError:
                pass

    @property
    def uses_inversion(self):
        return True

    @property
    def uses_hyper_images(self):
        return True

    class Analysis(phase_imaging.LensSourcePlanePhase.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            pass

        def __init__(self, lens_data, cosmology, positions_threshold, results=None,
                     uses_hyper_images=False):
            super(HyperPixelizationPhase.Analysis, self).__init__(
                lens_data=lens_data, cosmology=cosmology,
                positions_threshold=positions_threshold,
                results=results, uses_hyper_images=uses_hyper_images)


class HyperGalaxyPhase(phase_imaging.PhaseImaging, af.HyperPhase):

    class Analysis(af.Analysis):

        def __init__(self, lens_data, model_image_2d, galaxy_image_2d):
            """
            An analysis to fit the noise for a single galaxy image.
            Parameters
            ----------
            lens_data: LensData
                Lens data, including an image and noise
            model_image_2d: ndarray
                An image produce of the overall system by a model
            galaxy_image_2d: ndarray
                The contribution of one galaxy to the model image
            """
            self.lens_data = lens_data
            self.hyper_model_image_1d = lens_data.array_1d_from_array_2d(
                array_2d=model_image_2d)
            self.hyper_galaxy_image_1d = lens_data.array_1d_from_array_2d(
                array_2d=galaxy_image_2d)

            self.check_for_previously_masked_values(array=self.hyper_model_image_1d)
            self.check_for_previously_masked_values(array=self.hyper_galaxy_image_1d)

        @staticmethod
        def check_for_previously_masked_values(array):
            if not np.all(array) != 0.0:
                raise exc.PhaseException(
                    'When mapping a 2D array to a 1D array using lens data, a value '
                    'encountered was 0.0 and therefore masked in a previous phase.')

        def visualize(self, instance, image_path, during_analysis):
            pass

        def fit(self, instance):
            """
            Fit the model image to the real image by scaling the hyper noise.
            Parameters
            ----------
            instance: ModelInstance
                A model instance with a hyper galaxy property
            Returns
            -------
            fit: float
            """
            fit = self.fit_for_hyper_galaxy(hyper_galaxy=instance.hyper_galaxy)
            return fit.figure_of_merit

        def fit_for_hyper_galaxy(self, hyper_galaxy):
            hyper_noise_1d = (
                hyper_galaxy.hyper_noise_map_from_hyper_images_and_noise_map(
                    hyper_model_image=self.hyper_model_image_1d,
                    hyper_galaxy_image=self.hyper_galaxy_image_1d,
                    noise_map=self.lens_data.noise_map_1d
                )
            )

            hyper_noise_map_1d = self.lens_data.noise_map_1d + hyper_noise_1d
            return lens_fit.LensDataFit(
                image_1d=self.lens_data.image_1d,
                noise_map_1d=hyper_noise_map_1d,
                mask_1d=self.lens_data.mask_1d,
                model_image_1d=self.hyper_model_image_1d,
                map_to_scaled_array=self.lens_data.map_to_scaled_array
            )

        @classmethod
        def describe(cls, instance):
            return "Running hyper galaxy fit for HyperGalaxy:\n{}".format(
                instance.hyper_galaxy)

    def run(self, data, results=None, mask=None, positions=None):
        """
        Run a fit for each galaxy from the previous phase.
        Parameters
        ----------
        data: LensData
        results: ResultsCollection
            Results from all previous phases
        mask: Mask
            The mask
        positions
        Returns
        -------
        results: HyperGalaxyResults
            A collection of results, with one item per a galaxy
        """

        mask = setup_phase_mask(data=data, mask=mask, mask_function=self.mask_function,
                                inner_mask_radii=self.inner_mask_radii)

        lens_data = ld.LensData(ccd_data=data, mask=mask,
                                sub_grid_size=self.sub_grid_size,
                                image_psf_shape=self.image_psf_shape,
                                positions=positions,
                                interp_pixel_scale=self.interp_pixel_scale,
                                uses_inversion=self.uses_inversion)

        model_image_2d = results.last.most_likely_fit.model_image_2d

        hyper_result = copy.deepcopy(results.last)
        hyper_result.analysis.uses_hyper_images = True
        hyper_result.analysis.hyper_model_image_1d = lens_data.array_1d_from_array_2d(
            array_2d=model_image_2d)
        hyper_result.analysis.hyper_galaxy_image_1d_path_dict = {}

        for galaxy_path, galaxy in results.last.path_galaxy_tuples:

            optimizer = self.optimizer.copy_with_name_extension(
                extension=galaxy_path[-1])
            optimizer.variable.hyper_galaxy = g.HyperGalaxy
            galaxy_image_2d = results.last.image_2d_dict[galaxy_path]

            # If array is all zeros, galaxy did not have image in previous phase and
            # should be ignored
            if not np.all(galaxy_image_2d == 0):
                hyper_result.analysis.hyper_galaxy_image_1d_path_dict[
                    galaxy_path] = lens_data.array_1d_from_array_2d(
                    array_2d=galaxy_image_2d)
                analysis = self.__class__.Analysis(lens_data=lens_data,
                                                   model_image_2d=model_image_2d,
                                                   galaxy_image_2d=galaxy_image_2d)
                result = optimizer.fit(analysis)

                hyper_result.constant.object_for_path(
                    galaxy_path
                ).hyper_galaxy = result.constant.hyper_galaxy

        results.hyper = hyper_result

        return results
