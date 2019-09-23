from astropy import cosmology as cosmo

import autofit as af
from autolens import exc
from autolens.array import mask as msk
from autolens.lens import lens_fit
from autolens.model.inversion import pixelizations as pix
from autolens.pipeline.phase import phase_extensions
from autolens.pipeline.phase.phase import AbstractPhase


def default_mask_function(image):
    return msk.Mask.circular(
        shape=image.shape, pixel_scale=image.pixel_scale, sub_size=1, radius_arcsec=3.0
    )


def isinstance_or_prior(obj, cls):
    if isinstance(obj, cls):
        return True
    if isinstance(obj, af.PriorModel) and obj.cls == cls:
        return True
    return False


class PhaseData(AbstractPhase):
    galaxies = af.PhaseProperty("galaxies")

    def __init__(
        self,
        phase_name,
        phase_tag,
        phase_folders=tuple(),
        galaxies=None,
        optimizer_class=af.MultiNest,
        cosmology=cosmo.Planck15,
        sub_size=2,
        signal_to_noise_limit=None,
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
            If *True*, the hyper_galaxies image used to generate the cluster'grids weight map will be binned
            up to this higher pixel scale to speed up the KMeans clustering algorithm. \
        """

        super(PhaseData, self).__init__(
            phase_name=phase_name,
            phase_tag=phase_tag,
            phase_folders=phase_folders,
            optimizer_class=optimizer_class,
            cosmology=cosmology,
            auto_link_priors=auto_link_priors,
        )

        self.sub_size = sub_size
        self.signal_to_noise_limit = signal_to_noise_limit
        self.positions_threshold = positions_threshold
        self.mask_function = mask_function
        self.inner_mask_radii = inner_mask_radii
        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid
        self.pixel_scale_binned_cluster_grid = pixel_scale_binned_cluster_grid
        self.inversion_uses_border = inversion_uses_border

        self.inversion_pixel_limit = (
            inversion_pixel_limit
            or af.conf.instance.general.get(
                "inversion", "inversion_pixel_limit_overall", int
            )
        )
        self.hyper_noise_map_max = af.conf.instance.general.get(
            "hyper", "hyper_noise_map_max", float
        )

        self.galaxies = galaxies or []

        self.is_hyper_phase = False

    @property
    def pixelization(self):
        for galaxy in self.galaxies:
            if galaxy.pixelization is not None:
                if isinstance(galaxy.pixelization, af.PriorModel):
                    return galaxy.pixelization.cls
                else:
                    return galaxy.pixelization

    @property
    def uses_cluster_inversion(self):
        if self.galaxies:
            for galaxy in self.galaxies:
                if isinstance_or_prior(galaxy.pixelization, pix.VoronoiBrightnessImage):
                    return True
        return False

    def run(self, data, results=None, mask=None, positions=None):
        """
        Run this phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        data: scaled_array.ScaledSquarePixelArray
            An lens_data that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper_galaxies.
        """
        analysis = self.make_analysis(
            data=data, results=results, mask=mask, positions=positions
        )

        self.variable = self.variable.populate(results)
        self.customize_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, data, results=None, mask=None, positions=None):
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
        raise NotImplementedError()

    def setup_phase_mask(self, data, mask):

        if self.mask_function is not None:
            mask = self.mask_function(image=data.image, sub_size=self.sub_size)
        elif mask is None and self.mask_function is None:
            mask = default_mask_function(image=data.image)

        if mask.sub_size != self.sub_size:
            mask = mask.new_mask_with_new_sub_size(sub_size=self.sub_size)

        if self.inner_mask_radii is not None:
            inner_mask = msk.Mask.circular(
                shape=mask.shape,
                pixel_scale=mask.pixel_scale,
                radius_arcsec=self.inner_mask_radii,
                sub_size=self.sub_size,
                invert=True,
            )
            mask = mask + inner_mask

        return mask

    def check_positions(self, positions):

        if self.positions_threshold is not None and positions is None:
            raise exc.PhaseException(
                "You have specified for a phase to use positions, but not input positions to the "
                "pipeline when you ran it."
            )

    def pixel_scale_binned_grid_from_mask(self, mask):

        pixel_scale_binned_grid = None

        if self.uses_cluster_inversion:

            if self.pixel_scale_binned_cluster_grid is None:

                pixel_scale_binned_cluster_grid = mask.pixel_scale

            else:

                pixel_scale_binned_cluster_grid = self.pixel_scale_binned_cluster_grid

            if pixel_scale_binned_cluster_grid > mask.pixel_scale:

                bin_up_factor = int(
                    self.pixel_scale_binned_cluster_grid / mask.pixel_scale
                )

            else:

                bin_up_factor = 1

            binned_mask = mask.binned_up_mask_from_mask(bin_up_factor=bin_up_factor)

            while binned_mask.pixels_in_mask < self.inversion_pixel_limit:

                if bin_up_factor == 1:
                    raise exc.DataException(
                        f"The pixelization {self.pixelization} uses a KMeans clustering algorithm which uses "
                        f"a hyper model image to adapt the pixelization. This hyper model image must have "
                        f"more pixels than inversion pixels. Current, the inversion_pixel_limit exceeds the "
                        f"data-points in the image.\n\n To rectify this image, manually set the inversion "
                        f"pixel limit in the pipeline phases or change the inversion_pixel_limit_overall "
                        f"parameter in general.ini "
                    )

                bin_up_factor -= 1
                binned_mask = mask.binned_up_mask_from_mask(bin_up_factor=bin_up_factor)

            pixel_scale_binned_grid = mask.pixel_scale * bin_up_factor

        return pixel_scale_binned_grid

    def preload_pixelization_grids_of_planes_from_results(self, results):

        if self.is_hyper_phase:
            return None

        if (
            results is not None
            and results.last is not None
            and hasattr(results.last, "hyper_combined")
            and self.pixelization is not None
        ):
            if self.pixelization.__class__ is results.last.pixelization.__class__:
                return (
                    results.last.hyper_combined.most_likely_pixelization_grids_of_planes
                )
        return None

    def extend_with_inversion_phase(self):
        return phase_extensions.InversionPhase(phase=self)

    # noinspection PyAbstractClass
    class Analysis(AbstractPhase.Analysis):
        @property
        def lens_data(self):
            raise NotImplementedError()

        def check_positions_trace_within_threshold_via_tracer(self, tracer):

            if (
                self.lens_data.positions is not None
                and self.lens_data.positions_threshold is not None
            ):

                traced_positions_of_planes = tracer.traced_positions_of_planes_from_positions(
                    positions=self.lens_data.positions
                )

                fit = lens_fit.LensPositionFit(
                    positions=traced_positions_of_planes[-1],
                    noise_map=self.lens_data.pixel_scale,
                )

                if not fit.maximum_separation_within_threshold(
                    self.lens_data.positions_threshold
                ):
                    raise exc.RayTracingException

        def check_inversion_pixels_are_below_limit_via_tracer(self, tracer):

            if self.lens_data.inversion_pixel_limit is not None:
                pixelizations = list(filter(None, tracer.pixelizations_of_planes))
                if pixelizations:
                    for pixelization in pixelizations:
                        if pixelization.pixels > self.lens_data.inversion_pixel_limit:
                            raise exc.PixelizationException

    class Result(AbstractPhase.Result):
        @property
        def most_likely_fit(self):

            hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
                instance=self.constant
            )

            hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
                instance=self.constant
            )

            return self.analysis.lens_imaging_fit_for_tracer(
                tracer=self.most_likely_tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

        @property
        def mask(self):
            return self.most_likely_fit.mask

        @property
        def positions(self):
            return self.most_likely_fit.positions

        @property
        def pixelization(self):
            for galaxy in self.most_likely_fit.tracer.galaxies:
                if galaxy.pixelization is not None:
                    return galaxy.pixelization

        @property
        def most_likely_pixelization_grids_of_planes(self):
            return self.most_likely_tracer.pixelization_grids_of_planes_from_grid(
                grid=self.most_likely_fit.grid
            )[-1]
