from os import path
import autofit as af
from astropy import cosmology as cosmo
from autogalaxy.pipeline.phase import abstract
from autolens.pipeline.phase.settings import SettingsPhasePositions
from autolens.pipeline.phase.point_source.analysis import Analysis
from autolens.pipeline.phase.point_source.result import Result


class PhasePointSource(abstract.AbstractPhase):

    galaxies = af.PhaseProperty("galaxies")

    Analysis = Analysis
    Result = Result

    def __init__(
        self,
        *,
        search,
        positions_solver,
        galaxies=None,
        settings=SettingsPhasePositions(),
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

        super().__init__(
            search=search, settings=settings, galaxies=galaxies, cosmology=cosmology
        )

        self.positions_solver = positions_solver

    def make_analysis(
        self,
        positions,
        positions_noise_map,
        fluxes=None,
        fluxes_noise_map=None,
        imaging=None,
        results=None,
    ):
        """
        Returns an lens object. Also calls the prior passing and masked_imaging modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask2D
            The default masks passed in by the pipeline
        dataset: im.Imaging
            An masked_imaging that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the `NonLinearSearch` calls to determine the fit of a set of values
        """

        self.output_phase_info()

        return self.Analysis(
            positions=positions,
            noise_map=positions_noise_map,
            fluxes=fluxes,
            fluxes_noise_map=fluxes_noise_map,
            solver=self.positions_solver,
            imaging=imaging,
            settings=self.settings,
            cosmology=self.cosmology,
            results=results,
        )

    def run(
        self,
        positions,
        positions_noise_map,
        fluxes=None,
        fluxes_noise_map=None,
        imaging=None,
        results=None,
        info=None,
        pickle_files=None,
    ):
        """
        Run this phase.

        Parameters
        ----------
        mask: Mask2D
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

        self.model = self.model.populate(results)

        results = results or af.ResultsCollection()

        analysis = self.make_analysis(
            positions=positions,
            positions_noise_map=positions_noise_map,
            fluxes=fluxes,
            fluxes_noise_map=fluxes_noise_map,
            imaging=imaging,
            results=results,
        )

        result = self.run_analysis(
            analysis=analysis, info=info, pickle_files=pickle_files
        )

        return self.make_result(result=result, analysis=analysis)

    def output_phase_info(self):

        file_phase_info = path.join(self.search.paths.output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.search).__name__))
            phase_info.write("Cosmology = {} \n".format(self.cosmology))
            phase_info.close()
