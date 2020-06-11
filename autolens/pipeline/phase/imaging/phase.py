import autofit as af
from astropy import cosmology as cosmo
from autogalaxy.pipeline.phase import dataset
from autogalaxy.pipeline.phase.imaging.phase import PhaseAttributes as AgPhaseAttributes
from autolens.pipeline.phase.settings import PhaseSettingsImaging
from autolens.pipeline.phase.imaging.analysis import Analysis
from autolens.pipeline.phase.imaging.meta_imaging import MetaImaging
from autolens.pipeline.phase.imaging.result import Result


class PhaseImaging(dataset.PhaseDataset):

    galaxies = af.PhaseProperty("galaxies")
    hyper_image_sky = af.PhaseProperty("hyper_image_sky")
    hyper_background_noise = af.PhaseProperty("hyper_background_noise")

    Analysis = Analysis
    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        search,
        galaxies=None,
        hyper_image_sky=None,
        hyper_background_noise=None,
        settings=PhaseSettingsImaging(),
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
            paths,
            search=search,
            settings=settings,
            galaxies=galaxies,
            cosmology=cosmology,
        )

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.is_hyper_phase = False

        self.meta_dataset = MetaImaging(
            settings=settings, model=self.model, is_hyper_phase=False
        )

    def make_phase_attributes(self, analysis):
        return PhaseAttributes(
            cosmology=self.cosmology,
            positions=analysis.masked_dataset.positions,
            hyper_model_image=analysis.hyper_model_image,
            hyper_galaxy_image_path_dict=analysis.hyper_galaxy_image_path_dict,
        )

    def make_analysis(self, dataset, mask, results=None):
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
            An lens object that the non-linear search calls to determine the fit of a set of values
        """
        self.meta_dataset.model = self.model

        masked_imaging = self.meta_dataset.masked_dataset_from(
            dataset=dataset, mask=mask, results=results
        )

        self.output_phase_info()

        analysis = self.Analysis(
            masked_imaging=masked_imaging,
            cosmology=self.cosmology,
            image_path=self.search.paths.image_path,
            results=results,
            log_likelihood_cap=self.settings.log_likelihood_cap,
        )

        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.search.paths.output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.search).__name__))
            phase_info.write(
                "Sub-grid size = {} \n".format(self.meta_dataset.settings.sub_size)
            )
            phase_info.write(
                "PSF shape = {} \n".format(self.meta_dataset.settings.psf_shape_2d)
            )
            phase_info.write(
                "Positions Threshold = {} \n".format(
                    self.meta_dataset.settings.positions_threshold
                )
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))

            phase_info.close()


class PhaseAttributes(AgPhaseAttributes):
    def __init__(
        self, cosmology, positions, hyper_model_image, hyper_galaxy_image_path_dict
    ):
        super().__init__(
            cosmology=cosmology,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        )

        self.positions = positions
