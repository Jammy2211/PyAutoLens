import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autolens.lens import ray_tracing, lens_fit
from autolens.pipeline.phase.phase import AbstractPhase, Phase


class PhasePositions(AbstractPhase):
    lens_galaxies = af.PhaseProperty("lens_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies]

    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), lens_galaxies=None,
                 optimizer_class=af.MultiNest,
                 cosmology=cosmo.Planck15, auto_link_priors=False):
        super().__init__(phase_name=phase_name, phase_tag=None, phase_folders=phase_folders, tag_phases=tag_phases,
                         optimizer_class=optimizer_class, cosmology=cosmology, auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies

    def run(self, positions, pixel_scale, results=None):
        """
        Run this phase.

        Parameters
        ----------
        pixel_scale
        positions
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(positions=positions, pixel_scale=pixel_scale, results=results)

        self.pass_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)
        return self.make_result(result, analysis)

    def make_analysis(self, positions, pixel_scale, results=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        pixel_scale
        positions
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens: Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """

        analysis = self.Analysis(positions=positions, pixel_scale=pixel_scale, cosmology=self.cosmology,
                                           results=results)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, positions, pixel_scale, cosmology, results=None):
            super().__init__(cosmology=cosmology, results=results)

            self.positions = list(map(lambda position_set: np.asarray(position_set), positions))
            self.pixel_scale = pixel_scale

        def visualize(self, instance, image_path, during_analysis):
            pass

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
            tracer = self.tracer_for_instance(instance)
            fit = self.fit_for_tracer(tracer)
            return fit.figure_of_merit

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=instance.lens_galaxies,
                                                                image_plane_positions=self.positions,
                                                                cosmology=self.cosmology)

        def fit_for_tracer(self, tracer):
            return lens_fit.LensPositionFit(positions=tracer.source_plane.positions, noise_map=self.pixel_scale)

        @classmethod
        def describe(cls, instance):
            return "\nRunning lens lens for... \n\nLens Galaxy::\n{}\n\n".format(instance.lens_galaxies)