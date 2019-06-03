from astropy import cosmology as cosmo

from autofit.optimize import non_linear
from autofit.tools.phase_property import PhaseProperty
from autolens.data.plotters import ccd_plotters
from autolens.lens import ray_tracing, sensitivity_fit
from autolens.lens.plotters import ray_tracing_plotters, sensitivity_fit_plotters
from autolens.pipeline.phase.phase_imaging import PhaseImaging


class SensitivityPhase(PhaseImaging):
    lens_galaxies = PhaseProperty("lens_galaxies")
    source_galaxies = PhaseProperty("source_galaxies")
    sensitive_galaxies = PhaseProperty("sensitive_galaxies")

    def __init__(self, phase_name, tag_phases=None, phase_folders=None, lens_galaxies=None, source_galaxies=None,
                 sensitive_galaxies=None,
                 optimizer_class=non_linear.MultiNest, sub_grid_size=2, bin_up_factor=None, mask_function=None,
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

        super(SensitivityPhase, self).__init__(phase_name=phase_name, tag_phases=tag_phases,
                                               phase_folders=phase_folders,
                                               optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                                               bin_up_factor=bin_up_factor, mask_function=mask_function,
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
            fit = self.fit_for_tracers(tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)
            return fit.figure_of_merit

        def visualize(self, instance, image_path, during_analysis):
            self.plot_count += 1

            tracer_normal = self.tracer_normal_for_instance(instance)
            tracer_sensitive = self.tracer_sensitive_for_instance(instance)
            fit = self.fit_for_tracers(tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)

            ccd_plotters.plot_ccd_subplot(ccd_data=self.lens_data.ccd_data, mask=self.lens_data.mask,
                                          positions=self.lens_data.positions,
                                          output_path=image_path, output_format='png')

            ccd_plotters.plot_ccd_individual(ccd_data=self.lens_data.ccd_data, mask=self.lens_data.mask,
                                             positions=self.lens_data.positions,
                                             output_path=image_path,
                                             output_format='png')

            ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer_normal, output_path=image_path,
                                                          output_format='png', output_filename='tracer_normal')

            ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer_sensitive, output_path=image_path,
                                                          output_format='png', output_filename='tracer_sensitive')

            sensitivity_fit_plotters.plot_fit_subplot(fit=fit, output_path=image_path, output_format='png')

            return fit

        def tracer_normal_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(lens_galaxies=instance.lens_galaxies,
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
            return sensitivity_fit.fit_lens_data_with_sensitivity_tracers(lens_data=self.lens_data,
                                                                          tracer_normal=tracer_normal,
                                                                          tracer_sensitive=tracer_sensitive)

        @classmethod
        def describe(cls, instance):
            return "\nRunning lens/source lens for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n Sensitive " \
                   "Galaxy\n{}\n\n ".format(instance.lens_galaxies, instance.source_galaxies,
                                            instance.sensitive_galaxies)