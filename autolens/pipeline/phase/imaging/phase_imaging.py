from astropy import cosmology as cosmo

import autofit as af
from autolens.lens import lens_data as ld
from autolens.pipeline import phase_tagging
from autolens.pipeline.phase import data
from autolens.pipeline.phase import extensions
from autolens.pipeline.phase.imaging.analysis import Analysis
from autolens.pipeline.phase.imaging.result import Result


class MetaImagingFit(data.MetaDataFit):
    def __init__(
            self,
            variable,
            sub_size=2,
            is_hyper_phase=False,
            signal_to_noise_limit=None,
            positions_threshold=None,
            mask_function=None,
            inner_mask_radii=None,
            pixel_scale_interpolation_grid=None,
            pixel_scale_binned_cluster_grid=None,
            inversion_uses_border=True,
            inversion_pixel_limit=None,
            psf_shape=None,
            bin_up_factor=None
    ):
        super().__init__(
            variable=variable,
            sub_size=sub_size,
            is_hyper_phase=is_hyper_phase,
            signal_to_noise_limit=signal_to_noise_limit,
            positions_threshold=positions_threshold,
            mask_function=mask_function,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            pixel_scale_binned_cluster_grid=pixel_scale_binned_cluster_grid,
            inversion_uses_border=inversion_uses_border,
            inversion_pixel_limit=inversion_pixel_limit,
        )
        self.psf_shape = psf_shape
        self.bin_up_factor = bin_up_factor

    def data_fit_from(
            self,
            data,
            mask,
            positions,
            results,
            modified_image
    ):
        mask = self.setup_phase_mask(
            data=data,
            mask=mask
        )

        self.check_positions(positions=positions)

        if self.uses_cluster_inversion:
            pixel_scale_binned_grid = self.pixel_scale_binned_grid_from_mask(mask=mask)
        else:
            pixel_scale_binned_grid = None

        preload_pixelization_grids_of_planes = self.preload_pixelization_grids_of_planes_from_results(
            results=results
        )

        lens_imaging_data = ld.LensImagingData(
            imaging_data=data.new_imaging_data_with_modified_image(
                modified_image
            ),
            mask=mask,
            trimmed_psf_shape=self.psf_shape,
            positions=positions,
            positions_threshold=self.positions_threshold,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=pixel_scale_binned_grid,
            hyper_noise_map_max=self.hyper_noise_map_max,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            preload_pixelization_grids_of_planes=preload_pixelization_grids_of_planes,
        )

        if self.signal_to_noise_limit is not None:
            lens_imaging_data = lens_imaging_data.new_lens_imaging_data_with_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit
            )

        if self.bin_up_factor is not None:
            lens_imaging_data = lens_imaging_data.new_lens_imaging_data_with_binned_up_imaging_data_and_mask(
                bin_up_factor=self.bin_up_factor
            )

        return lens_imaging_data


class PhaseImaging(data.PhaseData):
    galaxies = af.PhaseProperty("galaxies")
    hyper_image_sky = af.PhaseProperty("hyper_image_sky")
    hyper_background_noise = af.PhaseProperty("hyper_background_noise")

    Analysis = Analysis
    Result = Result

    def __init__(
            self,
            phase_name,
            phase_folders=tuple(),
            galaxies=None,
            hyper_image_sky=None,
            hyper_background_noise=None,
            optimizer_class=af.MultiNest,
            cosmology=cosmo.Planck15,
            sub_size=2,
            signal_to_noise_limit=None,
            bin_up_factor=None,
            psf_shape=None,
            positions_threshold=None,
            mask_function=None,
            inner_mask_radii=None,
            pixel_scale_interpolation_grid=None,
            pixel_scale_binned_cluster_grid=None,
            inversion_uses_border=True,
            inversion_pixel_limit=None,
            auto_link_priors=False,
    ):

        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_size: int
            The side length of the subgrid
        pixel_scale_binned_cluster_grid : float or None
            If *True*, the hyper_galaxies image used to generate the cluster'grids weight map will be binned \
            up to this higher pixel scale to speed up the KMeans clustering algorithm.
        """

        phase_tag = phase_tagging.phase_tag_from_phase_settings(
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            psf_shape=psf_shape,
            positions_threshold=positions_threshold,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            pixel_scale_binned_cluster_grid=pixel_scale_binned_cluster_grid,
        )

        super().__init__(
            phase_name=phase_name,
            phase_tag=phase_tag,
            phase_folders=phase_folders,
            galaxies=galaxies,
            optimizer_class=optimizer_class,
            cosmology=cosmology,
            auto_link_priors=auto_link_priors,
        )

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.hyper_noise_map_max = af.conf.instance.general.get(
            "hyper", "hyper_noise_map_max", float
        )

        self.is_hyper_phase = False

        self.meta_data_fit = MetaImagingFit(
            variable=self.variable,
            bin_up_factor=bin_up_factor,
            psf_shape=psf_shape,
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
            positions_threshold=positions_threshold,
            mask_function=mask_function,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            pixel_scale_binned_cluster_grid=pixel_scale_binned_cluster_grid,
            inversion_uses_border=inversion_uses_border,
            inversion_pixel_limit=inversion_pixel_limit,
        )

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

    def make_analysis(
            self,
            data,
            results=None,
            mask=None,
            positions=None
    ):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        data: im.Imaging
            An lens_data that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """
        self.meta_data_fit.variable = self.variable
        modified_image = self.modify_image(
            image=data.image,
            results=results,
        )

        lens_imaging_data = self.meta_data_fit.data_fit_from(
            data=data,
            mask=mask,
            positions=positions,
            results=results,
            modified_image=modified_image
        )

        self.output_phase_info()

        analysis = self.Analysis(
            lens_imaging_data=lens_imaging_data,
            cosmology=self.cosmology,
            image_path=self.optimizer.image_path,
            results=results,
        )

        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.optimizer.phase_output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.optimizer).__name__))
            phase_info.write("Sub-grid size = {} \n".format(self.meta_data_fit.sub_size))
            phase_info.write("PSF shape = {} \n".format(self.meta_data_fit.psf_shape))
            phase_info.write(
                "Positions Threshold = {} \n".format(self.meta_data_fit.positions_threshold)
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))
            phase_info.write("Auto Link Priors = {} \n".format(self.auto_link_priors))

            phase_info.close()

    def extend_with_multiple_hyper_phases(
            self,
            hyper_galaxy=False,
            inversion=False,
            include_background_sky=False,
            include_background_noise=False,
    ):
        hyper_phase_classes = []

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

        if inversion:
            if not include_background_sky and not include_background_noise:
                hyper_phase_classes.append(
                    extensions.InversionPhase
                )
            elif include_background_sky and not include_background_noise:
                hyper_phase_classes.append(
                    extensions.InversionBackgroundSkyPhase
                )
            elif not include_background_sky and include_background_noise:
                hyper_phase_classes.append(
                    extensions.InversionBackgroundNoisePhase
                )
            else:
                hyper_phase_classes.append(
                    extensions.InversionBackgroundBothPhase
                )

        if len(hyper_phase_classes) == 0:
            return self
        else:
            return extensions.CombinedHyperPhase(
                phase=self,
                hyper_phase_classes=hyper_phase_classes
            )
