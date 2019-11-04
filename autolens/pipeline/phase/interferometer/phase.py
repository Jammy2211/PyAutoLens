from astropy import cosmology as cosmo

import autofit as af
from autolens.pipeline import phase_tagging
from autolens.pipeline.phase import data
from autolens.pipeline.phase import extensions
from autolens.pipeline.phase.interferometer.analysis import Analysis
from autolens.pipeline.phase.interferometer.meta_interferometer_fit import MetaInterferometerFit
from autolens.pipeline.phase.interferometer.result import Result


class PhaseInterferometer(data.PhaseData):
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
        primary_beam_shape_2d=None,
        positions_threshold=None,
        mask_function=None,
        inner_mask_radii=None,
        pixel_scale_interpolation_grid=None,
        inversion_uses_border=True,
        inversion_pixel_limit=None,
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
        """

        phase_tag = phase_tagging.phase_tag_from_phase_settings(
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
            primaary_beam_shape_2d=primary_beam_shape_2d,
            positions_threshold=positions_threshold,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        )

        super().__init__(
            phase_name=phase_name,
            phase_tag=phase_tag,
            phase_folders=phase_folders,
            galaxies=galaxies,
            optimizer_class=optimizer_class,
            cosmology=cosmology,
        )

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.is_hyper_phase = False

        self.meta_data_fit = MetaInterferometerFit(
            variable=self.variable,
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
            positions_threshold=positions_threshold,
            mask_function=mask_function,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_uses_border=inversion_uses_border,
            inversion_pixel_limit=inversion_pixel_limit,
        )

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def modify_visibilities(self, visibilities, results):
        """
        Customize an masked_interferometer. e.g. removing lens light.

        Parameters
        ----------
        image: scaled_array.ScaledSquarePixelArray
            An masked_interferometer that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous lens

        Returns
        -------
        masked_interferometer: scaled_array.ScaledSquarePixelArray
            The modified image (not changed by default)
        """
        return visibilities

    def make_analysis(self, data, results=None, mask=None, positions=None):
        """
        Create an lens object. Also calls the prior passing and masked_interferometer modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        data: im.Interferometer
            An masked_interferometer that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """
        self.meta_data_fit.variable = self.variable
        modified_visibilities = self.modify_visibilities(visibilities=data.visibilities, results=results)

        masked_interferometer = self.meta_data_fit.data_fit_from(
            data=data,
            mask=mask,
            positions=positions,
            results=results,
            modified_visibilities=modified_visibilities,
        )

        self.output_phase_info()

        analysis = self.Analysis(
            masked_interferometer=masked_interferometer,
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
            phase_info.write(
                "Sub-grid size = {} \n".format(self.meta_data_fit.sub_size)
            )
            phase_info.write("Primary Beam shape = {} \n".format(self.meta_data_fit.primary_beam_shape_2d))
            phase_info.write(
                "Positions Threshold = {} \n".format(
                    self.meta_data_fit.positions_threshold
                )
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))

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
                hyper_phase_classes.append(extensions.InversionPhase)
            elif include_background_sky and not include_background_noise:
                hyper_phase_classes.append(extensions.InversionBackgroundSkyPhase)
            elif not include_background_sky and include_background_noise:
                hyper_phase_classes.append(extensions.InversionBackgroundNoisePhase)
            else:
                hyper_phase_classes.append(extensions.InversionBackgroundBothPhase)

        if len(hyper_phase_classes) == 0:
            return self
        else:
            return extensions.CombinedHyperPhase(
                phase=self, hyper_phase_classes=hyper_phase_classes
            )
