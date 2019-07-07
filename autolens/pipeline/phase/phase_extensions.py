import copy

import numpy as np
from typing import cast
import os

import autofit as af
from autolens import exc
from autolens.lens import lens_data as ld, lens_fit
from autolens.model.galaxy import galaxy as g
from autolens.model.hyper import hyper_data as hd
from autolens.model.inversion import pixelizations as px
from autolens.model.inversion import regularization as rg
from autolens.pipeline.phase import phase as ph
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline.phase.phase import setup_phase_mask
from autolens.pipeline.plotters import hyper_plotters


class HyperPhase(object):
    def __init__(self, phase: ph.Phase):
        """
        Abstract HyperPhase. Wraps a regular phase, performing that phase before performing the action
        specified by the run_hyper.

        Parameters
        ----------
        phase
            A regular phase
        """
        self.phase = phase

    @property
    def hyper_name(self) -> str:
        """
        The name of the hyper form of the phase. This is used to generate folder names and also address the
        hyper results in the Result object.
        """
        raise NotImplementedError()

    def run_hyper(self, *args, **kwargs) -> af.Result:
        """
        Run the hyper phase.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        result
            The result of the hyper phase.
        """
        raise NotImplementedError()

    def make_hyper_phase(self) -> ph.Phase:
        """
        Returns
        -------
        hyper_phase
            A copy of the original phase with a modified name and path
        """

        phase = copy.deepcopy(self.phase)
        phase.phase_path = f"{phase.phase_path}/{phase.phase_name}"
  #      phase.phase_name = self.hyper_name

        phase_folders = phase.phase_folders
        phase_folders.append(phase.phase_name)

        phase.optimizer = af.MultiNest(
            phase_name=self.hyper_name,
            phase_tag=phase.phase_tag[8:], # Hack to remove first 'settngs'
            phase_folders=phase_folders,
            model_mapper=phase.optimizer.variable,
            sigma_limit=phase.optimizer.sigma_limit)

        return phase

    def run(self, data, results: af.ResultsCollection = None, **kwargs) -> af.Result:
        """
        Run the normal phase and then the hyper phase.

        Parameters
        ----------
        data
            Data
        results
            Results from previous phases.
        kwargs

        Returns
        -------
        result
            The result of the phase, with a hyper result attached as an attribute with the hyper_name of this
            phase.
        """
        results = copy.deepcopy(results) if results is not None else af.ResultsCollection()
        result = self.phase.run(data,results=results,**kwargs)
        results.add(self.phase.phase_name, result)
        hyper_result = self.run_hyper(
            data=data,
            results=results,
            **kwargs
        )
        setattr(result, self.hyper_name, hyper_result)
        return result


# noinspection PyAbstractClass
class VariableFixingHyperPhase(HyperPhase):

    def __init__(self, phase: ph.Phase, variable_classes=tuple()):

        super().__init__(phase)

        self.variable_classes = variable_classes

    def run_hyper(self, data, results=None, **kwargs):
        """
        Run the phase, overriding the optimizer's variable instance with one created to
        only fit pixelization hyperparameters.
        """

        variable = copy.deepcopy(results.last.variable)
        self.transfer_classes(results.last.constant, variable)

        phase = self.make_hyper_phase()
        phase.optimizer.variable = variable

        phase.const_efficiency_mode = \
            af.conf.instance.non_linear.get('MultiNest', 'extension_inversion_const_efficiency_mode', bool)

        phase.optimizer.sampling_efficiency = \
            af.conf.instance.non_linear.get('MultiNest', 'extension_inversion_sampling_efficiency', float)

        phase.optimizer.n_live_points = \
            af.conf.instance.non_linear.get('MultiNest', 'extension_inversion_n_live_points', int)


        return phase.run(data, results=results, **kwargs)

    def transfer_classes(self, instance, mapper):
        """
        Recursively overwrite priors in the mapper with constant values from the
        instance except where the containing class is the decedent of a listed class.

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
                if not any(
                        isinstance(
                            instance_value,
                            cls
                        )
                        for cls in self.variable_classes
                ):
                    try:
                        self.transfer_classes(
                            instance_value,
                            mapper_value)
                    except AttributeError:
                        setattr(mapper, key, instance_value)
            except AttributeError:
                pass


class InversionPhase(VariableFixingHyperPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: ph.Phase):

            super().__init__(
                phase=phase,
                variable_classes=(px.Pixelization,
                                  rg.Regularization))

    @property
    def hyper_name(self):
        return "inversion"

    @property
    def uses_inversion(self):
        return True

    @property
    def uses_hyper_images(self):
        return True

    class Analysis(phase_imaging.LensSourcePlanePhase.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            pass

        def __init__(self, lens_data, cosmology, positions_threshold, results=None):

            super(InversionPhase.Analysis, self).__init__(
                lens_data=lens_data, cosmology=cosmology,
                positions_threshold=positions_threshold,
                results=results)


# TODO : OBviously this isn't how we reallly want to implement this...

class InversionBackgroundSkyPhase(VariableFixingHyperPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: ph.Phase):

            super().__init__(
                phase=phase,
                variable_classes=(px.Pixelization,
                                  rg.Regularization,
                                  hd.HyperImageSky))

    @property
    def hyper_name(self):
        return "inversion"

    @property
    def uses_inversion(self):
        return True

    @property
    def uses_hyper_images(self):
        return True

    class Analysis(phase_imaging.LensSourcePlanePhase.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            pass

        def __init__(self, lens_data, cosmology, positions_threshold, results=None):

            super(InversionBackgroundSkyPhase.Analysis, self).__init__(
                lens_data=lens_data, cosmology=cosmology,
                positions_threshold=positions_threshold,
                results=results)


class InversionBackgroundNoisePhase(VariableFixingHyperPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: ph.Phase):

            super().__init__(
                phase=phase,
                variable_classes=(px.Pixelization,
                                  rg.Regularization,
                                  hd.HyperNoiseBackground))

    @property
    def hyper_name(self):
        return "inversion"

    @property
    def uses_inversion(self):
        return True

    @property
    def uses_hyper_images(self):
        return True

    class Analysis(phase_imaging.LensSourcePlanePhase.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            pass

        def __init__(self, lens_data, cosmology, positions_threshold, results=None):

            super(InversionBackgroundNoisePhase.Analysis, self).__init__(
                lens_data=lens_data, cosmology=cosmology,
                positions_threshold=positions_threshold,
                results=results)


class InversionBackgroundBothPhase(VariableFixingHyperPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: ph.Phase):

            super().__init__(
                phase=phase,
                variable_classes=(px.Pixelization,
                                  rg.Regularization,
                                  hd.HyperImageSky,
                                  hd.HyperNoiseBackground))

    @property
    def hyper_name(self):
        return "inversion"

    @property
    def uses_inversion(self):
        return True

    @property
    def uses_hyper_images(self):
        return True

    class Analysis(phase_imaging.LensSourcePlanePhase.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            pass

        def __init__(self, lens_data, cosmology, positions_threshold, results=None):

            super(InversionBackgroundBothPhase.Analysis, self).__init__(
                lens_data=lens_data, cosmology=cosmology,
                positions_threshold=positions_threshold,
                results=results)


class HyperGalaxyPhase(HyperPhase):

    @property
    def hyper_name(self):
        return "hyper_galaxy"

    class Analysis(af.Analysis):

        def __init__(self, lens_data, model_image_1d, galaxy_image_1d):
            """
            An analysis to fit the noise for a single galaxy image.
            Parameters
            ----------
            lens_data: LensData
                Lens data, including an image and noise
            model_image_1d: ndarray
                An image produce of the overall system by a model
            galaxy_image_1d: ndarray
                The contribution of one galaxy to the model image
            """

            self.lens_data = lens_data

            self.hyper_model_image_1d = model_image_1d
            self.hyper_galaxy_image_1d = galaxy_image_1d

            self.check_for_previously_masked_values(array=self.hyper_model_image_1d)
            self.check_for_previously_masked_values(array=self.hyper_galaxy_image_1d)

            self.plot_hyper_galaxy_subplot = \
                af.conf.instance.visualize.get('plots', 'plot_hyper_galaxy_subplot',
                                               bool)

        @staticmethod
        def check_for_previously_masked_values(array):
            if not np.all(array) != 0.0:
                raise exc.PhaseException(
                    'When mapping a 2D array to a 1D array using lens data, a value '
                    'encountered was 0.0 and therefore masked in a previous phase.')

        def visualize(self, instance, image_path, during_analysis):

            if self.plot_hyper_galaxy_subplot:

                hyper_model_image_2d = self.lens_data.scaled_array_2d_from_array_1d(
                    array_1d=self.hyper_model_image_1d)
                hyper_galaxy_image_2d = self.lens_data.scaled_array_2d_from_array_1d(
                    array_1d=self.hyper_galaxy_image_1d)

                hyper_image_sky = self.hyper_image_sky_for_instance(
                    instance=instance)

                hyper_noise_background = self.hyper_noise_background_for_instance(
                    instance=instance)

                hyper_galaxy = instance.hyper_galaxy

                contribution_map_2d = hyper_galaxy.contribution_map_from_hyper_images(
                    hyper_model_image=hyper_model_image_2d,
                    hyper_galaxy_image=hyper_galaxy_image_2d)

                fit_normal = lens_fit.LensDataFit(
                    image_1d=self.lens_data.image_1d,
                    noise_map_1d=self.lens_data.noise_map_1d,
                    mask_1d=self.lens_data.mask_1d,
                    model_image_1d=self.hyper_model_image_1d,
                    scaled_array_2d_from_array_1d=self.lens_data.scaled_array_2d_from_array_1d)

                fit = self.fit_for_hyper_galaxy(
                    hyper_galaxy=hyper_galaxy,
                    hyper_image_sky=hyper_image_sky,
                    hyper_noise_background=hyper_noise_background)

                hyper_plotters.plot_hyper_galaxy_subplot(
                    hyper_galaxy_image=hyper_galaxy_image_2d,
                    contribution_map=contribution_map_2d,
                    noise_map=self.lens_data.noise_map_2d,
                    hyper_noise_map=fit.noise_map_2d,
                    chi_squared_map=fit_normal.chi_squared_map_2d,
                    hyper_chi_squared_map=fit.chi_squared_map_2d,
                    output_path=image_path, output_format='png')

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

            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_noise_background = self.hyper_noise_background_for_instance(instance=instance)

            fit = self.fit_for_hyper_galaxy(
                hyper_galaxy=instance.hyper_galaxy,
                hyper_image_sky=hyper_image_sky,
                hyper_noise_background=hyper_noise_background)

            return fit.figure_of_merit

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

        def fit_for_hyper_galaxy(self, hyper_galaxy, hyper_image_sky, hyper_noise_background):

            if hyper_image_sky is not None:
                image_1d = hyper_image_sky.image_scaled_sky_from_image(image=self.lens_data.image_1d)
            else:
                image_1d = self.lens_data.image_1d

            if hyper_noise_background is not None:
                noise_map_1d = hyper_noise_background.noise_map_scaled_noise_from_noise_map(
                    noise_map=self.lens_data.noise_map_1d)
            else:
                noise_map_1d = self.lens_data.noise_map_1d

            hyper_noise_1d = hyper_galaxy.hyper_noise_map_from_hyper_images_and_noise_map(
                hyper_model_image=self.hyper_model_image_1d,
                hyper_galaxy_image=self.hyper_galaxy_image_1d,
                noise_map=self.lens_data.noise_map_1d)

            hyper_noise_map_1d = noise_map_1d + hyper_noise_1d

            return lens_fit.LensDataFit(
                image_1d=image_1d,
                noise_map_1d=hyper_noise_map_1d,
                mask_1d=self.lens_data.mask_1d,
                model_image_1d=self.hyper_model_image_1d,
                scaled_array_2d_from_array_1d=self.lens_data.scaled_array_2d_from_array_1d)

        @classmethod
        def describe(cls, instance):
            return "Running hyper galaxy fit for HyperGalaxy:\n{}".format(
                instance.hyper_galaxy)

    def run_hyper(self, data, results=None, mask=None, positions=None,
                  include_sky_background=False, include_noise_background=False):
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
        phase = self.make_hyper_phase()

        mask = setup_phase_mask(
            data=data,
            mask=mask,
            mask_function=cast(phase_imaging.PhaseImaging, phase).mask_function,
            inner_mask_radii=cast(phase_imaging.PhaseImaging, phase).inner_mask_radii
        )

        lens_data = ld.LensData(
            ccd_data=data,
            mask=mask,
            sub_grid_size=cast(phase_imaging.PhaseImaging, phase).sub_grid_size,
            image_psf_shape=cast(phase_imaging.PhaseImaging, phase).image_psf_shape,
            positions=positions,
            interp_pixel_scale=cast(phase_imaging.PhaseImaging, phase).interp_pixel_scale,
            cluster_pixel_scale=cast(phase_imaging.PhaseImaging, phase).cluster_pixel_scale,
            cluster_pixel_limit=cast(phase_imaging.PhaseImaging, phase).cluster_pixel_limit,
            uses_inversion=cast(phase_imaging.PhaseImaging, phase).uses_inversion,
            uses_cluster_inversion=cast(phase_imaging.PhaseImaging, phase).uses_cluster_inversion
        )

        model_image_1d = results.last.hyper_model_image_1d_from_mask(mask=lens_data.mask_2d)
        hyper_galaxy_image_1d_path_dict = \
            results.last.hyper_galaxy_image_1d_path_dict_from_mask(mask=lens_data.mask_2d)

        hyper_result = copy.deepcopy(results.last)
        hyper_result.analysis.uses_hyper_images = True
        hyper_result.analysis.hyper_model_image_1d = model_image_1d
        hyper_result.analysis.hyper_galaxy_image_1d_path_dict = hyper_galaxy_image_1d_path_dict

        for path, galaxy in results.last.path_galaxy_tuples:

            optimizer = phase.optimizer.copy_with_name_extension(
                extension=path[-1])

            # TODO : This is a HACK :O

            optimizer.variable.lens_galaxies = []
            optimizer.variable.source_galaxies = []
            optimizer.variable.galaxies = []

            phase.const_efficiency_mode = \
                af.conf.instance.non_linear.get('MultiNest', 'extension_hyper_galaxy_const_efficiency_mode', bool)

            phase.optimizer.sampling_efficiency = \
                af.conf.instance.non_linear.get('MultiNest', 'extension_hyper_galaxy_sampling_efficiency', float)

            phase.optimizer.n_live_points = \
                af.conf.instance.non_linear.get('MultiNest', 'extension_hyper_galaxy_n_live_points', int)

            optimizer.variable.hyper_galaxy = g.HyperGalaxy

            if include_sky_background:
                optimizer.variable.hyper_image_sky = hd.HyperImageSky

            if include_noise_background:
                optimizer.variable.hyper_noise_background = hd.HyperNoiseBackground

            # If array is all zeros, galaxy did not have image in previous phase and
            # should be ignored
            if not np.all(hyper_galaxy_image_1d_path_dict[path] == 0):

                analysis = self.Analysis(
                    lens_data=lens_data,
                    model_image_1d=model_image_1d,
                    galaxy_image_1d=hyper_galaxy_image_1d_path_dict[path])

                result = optimizer.fit(analysis)

                hyper_result.constant.object_for_path(path).hyper_galaxy = result.constant.hyper_galaxy

                if include_sky_background:
                    hyper_result.constant.object_for_path(path).hyper_image_sky = result.constant.hyper_image_sky

                if include_noise_background:
                    hyper_result.constant.object_for_path(path).hyper_noise_background = result.constant.hyper_noise_background

        return hyper_result


class HyperGalaxyBackgroundSkyPhase(HyperGalaxyPhase):

    def run_hyper(self, data, results=None, mask=None, positions=None,
                  include_sky_background=True, include_noise_background=False):

        return super().run_hyper(data=data,
                                 results=results,
                                 mask=mask,
                                 positions=positions,
                                 include_sky_background=True,
                                 include_noise_background=False)


class HyperGalaxyBackgroundNoisePhase(HyperGalaxyPhase):

    def run_hyper(self, data, results=None, mask=None, positions=None,
                  include_sky_background=False, include_noise_background=True):

        return super().run_hyper(data=data,
                                 results=results,
                                 mask=mask,
                                 positions=positions,
                                 include_sky_background=False,
                                 include_noise_background=True)


class HyperGalaxyBackgroundBoth(HyperGalaxyPhase):

    def run_hyper(self, data, results=None, mask=None, positions=None,
                  include_sky_background=True, include_noise_background=True):

        return super().run_hyper(data=data,
                                 results=results,
                                 mask=mask,
                                 positions=positions,
                                 include_sky_background=True,
                                 include_noise_background=True)


class CombinedHyperPhase(phase_imaging.PhaseImaging):

    def __init__(self, phase: phase_imaging.PhaseImaging, hyper_phase_classes: (type,) = tuple()):
        """
        A combined hyper phase that can run zero or more other hyper phases after the initial phase is run.

        Parameters
        ----------
        phase : phase_imaging.PhaseImaging
            The phase wrapped by this hyper phase
        hyper_phase_classes
            The classes of hyper phases to be run following the initial phase
        """

        super().__init__(
            phase_name=phase.phase_name,
            phase_folders=phase.phase_folders,
            tag_phases=phase.tag_phases,
            optimizer_class=af.MultiNest,
            sub_grid_size=phase.sub_grid_size,
            bin_up_factor=phase.bin_up_factor,
            image_psf_shape=phase.image_psf_shape,
            inversion_psf_shape=phase.inversion_psf_shape,
            positions_threshold=phase.positions_threshold,
            mask_function=phase.mask_function,
            inner_mask_radii=phase.inner_mask_radii,
            interp_pixel_scale=phase.interp_pixel_scale,
            inversion_pixel_limit=phase.inversion_pixel_limit,
            cluster_pixel_scale=phase.cluster_pixel_scale,
            cosmology=phase.cosmology,
            auto_link_priors=phase.auto_link_priors)

        self.hyper_phases = list(map(lambda cls: cls(phase),hyper_phase_classes))
        self.phase = phase

    def run(self, data, results: af.ResultsCollection = None, **kwargs) -> af.Result:
        """
        Run the regular phase followed by the hyper phases. Each result of a hyper phase is attached to the
        overall result object by the hyper_name of that phase.

        Parameters
        ----------
        data
            The data
        results
            Results from previous phases
        kwargs

        Returns
        -------
        result
            The result of the regular phase, with hyper results attached by associated hyper names
        """
        results = copy.deepcopy(results) if results is not None else af.ResultsCollection()
        result = self.phase.run(data, results=results, **kwargs)
        results.add(self.phase.phase_name, result)

        for hyper_phase in self.hyper_phases:
            hyper_result = hyper_phase.run_hyper(
                data=data,
                results=results,
                **kwargs
            )
            setattr(result, hyper_phase.hyper_name, hyper_result)
        return result
