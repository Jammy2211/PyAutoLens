from astropy import cosmology as cosmo

import autofit as af
from autolens.pipeline import tagging
from autolens.pipeline.phase import dataset
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
        galaxies=None,
        hyper_image_sky=None,
        hyper_background_noise=None,
        non_linear_class=af.MultiNest,
        cosmology=cosmo.Planck15,
        sub_size=2,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        psf_shape_2d=None,
        positions_threshold=None,
        pixel_scale_interpolation_grid=None,
        inversion_uses_border=True,
        inversion_pixel_limit=None,
    ):

        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        non_linear_class: class
            The class of a non_linear optimizer
        sub_size: int
            The side length of the subgrid
        """

        phase_tag = tagging.phase_tag_from_phase_settings(
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            psf_shape_2d=psf_shape_2d,
            positions_threshold=positions_threshold,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        )
        paths.phase_tag = phase_tag

        super().__init__(
            paths,
            galaxies=galaxies,
            non_linear_class=non_linear_class,
            cosmology=cosmology,
        )

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.is_hyper_phase = False

        self.meta_dataset = MetaImaging(
            model=self.model,
            bin_up_factor=bin_up_factor,
            psf_shape_2d=psf_shape_2d,
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
            positions_threshold=positions_threshold,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_uses_border=inversion_uses_border,
            inversion_pixel_limit=inversion_pixel_limit,
        )

    def make_phase_attributes(self, analysis):
        return PhaseAttributes(
            cosmology=self.cosmology,
            hyper_model_image=analysis.hyper_model_image,
            hyper_galaxy_image_path_dict=analysis.hyper_galaxy_image_path_dict,
        )

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
        self.meta_dataset.model = self.model

        masked_imaging = self.meta_dataset.masked_dataset_from(
            dataset=dataset, mask=mask, positions=positions, results=results
        )

        self.output_phase_info()

        analysis = self.Analysis(
            masked_imaging=masked_imaging,
            cosmology=self.cosmology,
            image_path=self.optimizer.paths.image_path,
            results=results,
        )

        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(
            self.optimizer.paths.phase_output_path, "phase.info"
        )

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.optimizer).__name__))
            phase_info.write("Sub-grid size = {} \n".format(self.meta_dataset.sub_size))
            phase_info.write("PSF shape = {} \n".format(self.meta_dataset.psf_shape_2d))
            phase_info.write(
                "Positions Threshold = {} \n".format(
                    self.meta_dataset.positions_threshold
                )
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))

            phase_info.close()


class PhaseAttributes:
    def __init__(self, cosmology, hyper_model_image, hyper_galaxy_image_path_dict):

        self.cosmology = cosmology
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict
