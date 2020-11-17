from os import path
import autofit as af
from astropy import cosmology as cosmo
from autolens.dataset import interferometer
from autolens.pipeline.phase import dataset
from autolens.pipeline.phase.settings import SettingsPhaseInterferometer
from autolens.pipeline.phase.interferometer.analysis import Analysis
from autolens.pipeline.phase.interferometer.result import Result


class PhaseInterferometer(dataset.PhaseDataset):
    galaxies = af.PhaseProperty("galaxies")
    hyper_background_noise = af.PhaseProperty("hyper_background_noise")

    Analysis = Analysis
    Result = Result

    def __init__(
        self,
        *,
        search,
        real_space_mask,
        galaxies=None,
        hyper_background_noise=None,
        settings=SettingsPhaseInterferometer(),
        cosmology=cosmo.Planck15,
    ):

        """

        A phase in an lens pipeline. Uses the set non_linear search to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        sub_size: int
            The side length of the subgrid
        """

        search.paths.tag = settings.phase_tag_with_inversion

        super().__init__(
            search=search, galaxies=galaxies, settings=settings, cosmology=cosmology
        )

        self.hyper_background_noise = hyper_background_noise

        self.is_hyper_phase = False

        self.real_space_mask = real_space_mask

    def make_analysis(self, dataset, mask, results=None):
        """
        Returns an lens object. Also calls the prior passing and masked_interferometer modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask2D
            The default masks passed in by the pipeline
        dataset: im.Interferometer
            An masked_interferometer that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the `NonLinearSearch` calls to determine the fit of a set of values
        """

        masked_interferometer = interferometer.MaskedInterferometer(
            interferometer=dataset,
            visibilities_mask=mask,
            real_space_mask=self.real_space_mask,
            settings=self.settings.settings_masked_interferometer,
        )

        self.output_phase_info()

        return self.Analysis(
            masked_interferometer=masked_interferometer,
            settings=self.settings,
            cosmology=self.cosmology,
            results=results,
        )

    def output_phase_info(self):

        file_phase_info = path.join(self.search.paths.output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.search).__name__))
            phase_info.write(
                "Sub-grid size = {} \n".format(
                    self.settings.settings_masked_interferometer.sub_size
                )
            )
            phase_info.write(
                "Positions Threshold = {} \n".format(
                    self.settings.settings_lens.positions_threshold
                )
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))

            phase_info.close()
