from astropy import cosmology as cosmo

import autofit as af
from autofit.tools.phase import Dataset
from autolens.pipeline.phase import abstract
from autolens.pipeline.phase import extensions
from autolens.pipeline.phase.dataset.result import Result

import pickle


class PhaseDataset(abstract.AbstractPhase):
    galaxies = af.PhaseProperty("galaxies")

    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        galaxies=None,
        non_linear_class=af.MultiNest,
        cosmology=cosmo.Planck15,
    ):
        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        non_linear_class: class
            The class of a non_linear optimizer
        """

        super().__init__(paths, non_linear_class=non_linear_class)
        self.galaxies = galaxies or []
        self.cosmology = cosmology

        self.is_hyper_phase = False

    def run(self, dataset: Dataset, mask, results=None, positions=None):
        """
        Run this phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        dataset: scaled_array.ScaledSquarePixelArray
            An masked_imaging that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper_galaxies.
        """
        dataset.save(self.paths.phase_output_path)
        self.save_metadata(dataset)
        self.save_mask(mask)
        self.save_meta_dataset(meta_dataset=self.meta_dataset)

        self.model = self.model.populate(results)

        analysis = self.make_analysis(
            dataset=dataset, mask=mask, results=results, positions=positions
        )

        phase_attributes = self.make_phase_attributes(analysis=analysis)
        self.save_phase_attributes(phase_attributes=phase_attributes)

        self.customize_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, mask, results=None, positions=None):
        """
        Create an lens object. Also calls the prior passing and masked_imaging modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        dataset: im.Imaging
            An masked_imaging that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """
        raise NotImplementedError()

    def extend_with_inversion_phase(self):
        return extensions.InversionPhase(phase=self)

    def extend_with_multiple_hyper_phases(
        self,
        hyper_galaxy=False,
        inversion=False,
        include_background_sky=False,
        include_background_noise=False,
        hyper_galaxy_phase_first=False,
    ):

        self.use_as_hyper_dataset = True

        hyper_phase_classes = []

        if self.meta_dataset.has_pixelization and inversion:
            if not include_background_sky and not include_background_noise:
                hyper_phase_classes.append(extensions.InversionPhase)
            elif include_background_sky and not include_background_noise:
                hyper_phase_classes.append(extensions.InversionBackgroundSkyPhase)
            elif not include_background_sky and include_background_noise:
                hyper_phase_classes.append(extensions.InversionBackgroundNoisePhase)
            else:
                hyper_phase_classes.append(extensions.InversionBackgroundBothPhase)

        if hyper_galaxy:
            if not include_background_sky and not include_background_noise:
                hyper_phase_classes.append(
                    extensions.hyper_galaxy_phase.HyperGalaxyPhase
                )
            elif include_background_sky and not include_background_noise:
                hyper_phase_classes.append(
                    extensions.hyper_galaxy_phase.HyperGalaxyBackgroundSkyPhase
                )
            elif not include_background_sky and include_background_noise:
                hyper_phase_classes.append(
                    extensions.hyper_galaxy_phase.HyperGalaxyBackgroundNoisePhase
                )
            else:
                hyper_phase_classes.append(
                    extensions.hyper_galaxy_phase.HyperGalaxyBackgroundBothPhase
                )

        if hyper_galaxy_phase_first:
            if inversion and hyper_galaxy:
                hyper_phase_classes = [cls for cls in reversed(hyper_phase_classes)]

        if len(hyper_phase_classes) == 0:
            return self
        else:
            return extensions.CombinedHyperPhase(
                phase=self, hyper_phase_classes=hyper_phase_classes
            )
