from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg
from autolens.pipeline import setup
from autolens import exc


class SLAM:
    def __init__(self, hyper=None, source=None, light=None, mass=None):

        self.hyper = hyper
        self.source = source
        self.light = light
        self.mass = mass

    def set_source_type(self, source_type):

        self.source.type_tag = source_type

    def set_light_type(self, light_type):

        self.light.type_tag = light_type

    def set_mass_type(self, mass_type):

        self.mass.type_tag = mass_type


class Hyper(setup.PipelineSetup):
    def __init__(
        self,
        hyper_galaxies=False,
        hyper_image_sky=False,
        hyper_background_noise=False,
        hyper_fixed_after_source=False,
    ):
        """The setup of a pipeline, which controls how PyAutoLens template pipelines runs, for example controlling
        assumptions about the lens's light profile bulge-disk model or if shear is included in the mass model.

        Users can write their own pipelines which do not use or require the *PipelineSetup* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        hyper_galaxies : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used to scale the
            noise-map of the dataset throughout the fitting.
        hyper_image_sky : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            image's background sky component in the model.
        hyper_background_noise : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            noise-map's background component in the model.
        hyper_fixed_after_source : bool
            If, after the Source pipeline, the hyper parameters are fixed and thus no longer reopitmized using new
            mass models.
        """

        super().__init__(
            hyper_galaxies=hyper_galaxies,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        self.hyper_fixed_after_source = hyper_fixed_after_source

    @property
    def hyper_tag(self):
        """Tag ithe hyper pipeline features used in a hyper pipeline to customize pipeline output paths.
        """
        if not any(
            [self.hyper_galaxies, self.hyper_image_sky, self.hyper_background_noise]
        ):
            return ""

        return (
            "__hyper"
            + self.hyper_galaxies_tag
            + self.hyper_image_sky_tag
            + self.hyper_background_noise_tag
            + self.hyper_fixed_after_source_tag
        )

    @property
    def hyper_fixed_after_source_tag(self):
        """Generate a tag for if the hyper parameters are held fixed after the source pipeline.

        This changes the pipeline setup tag as follows:

        hyper_fixed_after_source = False -> setup
        hyper_fixed_after_source = True -> setup__hyper_fixed
        """
        if not self.hyper_fixed_after_source:
            return ""
        elif self.hyper_fixed_after_source:
            return "_fixed"


class Source(setup.PipelineSetup):
    def __init__(
        self,
        pixelization=pix.VoronoiBrightnessImage,
        regularization=reg.AdaptiveBrightness,
        no_shear=False,
        lens_light_centre=None,
        lens_mass_centre=None,
        align_light_mass_centre=False,
        lens_light_bulge_only=False,
        number_of_gaussians=None,
        fix_lens_light=False,
    ):

        super().__init__(
            pixelization=pixelization,
            regularization=regularization,
            no_shear=no_shear,
            lens_light_centre=lens_light_centre,
            lens_mass_centre=lens_mass_centre,
            number_of_gaussians=number_of_gaussians,
            align_light_mass_centre=align_light_mass_centre,
        )

        self.lens_light_bulge_only = lens_light_bulge_only
        self.fix_lens_light = fix_lens_light
        self.type_tag = None

    @property
    def tag(self):
        return (
            "source__"
            + self.type_tag
            + self.number_of_gaussians_tag
            + self.no_shear_tag
            + self.lens_light_centre_tag
            + self.lens_mass_centre_tag
            + self.align_light_mass_centre_tag
            + self.lens_light_bulge_only_tag
            + self.fix_lens_light_tag
        )

    @property
    def lens_light_bulge_only_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the setup folder as follows:

        fix_lens_light = False -> setup__
        fix_lens_light = True -> setup___fix_lens_light
        """
        if not self.lens_light_bulge_only:
            return ""
        elif self.lens_light_bulge_only:
            return "__bulge_only"

    @property
    def fix_lens_light_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the setup folder as follows:

        fix_lens_light = False -> setup__
        fix_lens_light = True -> setup___fix_lens_light
        """
        if not self.fix_lens_light:
            return ""
        elif self.fix_lens_light:
            return "__fix_lens_light"


class Light(setup.PipelineSetup):
    def __init__(
        self,
        align_bulge_disk_centre=False,
        align_bulge_disk_phi=False,
        align_bulge_disk_axis_ratio=False,
        disk_as_sersic=False,
        number_of_gaussians=None,
    ):

        super().__init__(
            align_bulge_disk_centre=align_bulge_disk_centre,
            align_bulge_disk_phi=align_bulge_disk_phi,
            align_bulge_disk_axis_ratio=align_bulge_disk_axis_ratio,
            disk_as_sersic=disk_as_sersic,
            number_of_gaussians=number_of_gaussians,
        )

        self.type_tag = None

    @property
    def tag(self):
        if self.number_of_gaussians is None:
            return (
                "light__"
                + self.type_tag
                + self.align_bulge_disk_tag
                + self.disk_as_sersic_tag
            )
        else:
            return "light__" + self.type_tag + self.number_of_gaussians_tag


class Mass(setup.PipelineSetup):
    def __init__(
        self,
        no_shear=False,
        align_light_dark_centre=False,
        align_bulge_dark_centre=False,
        fix_lens_light=False,
    ):

        super().__init__(
            no_shear=no_shear,
            align_light_dark_centre=align_light_dark_centre,
            align_bulge_dark_centre=align_bulge_dark_centre,
        )

        self.fix_lens_light = fix_lens_light
        self.type_tag = None

    @property
    def tag(self):
        return (
            "mass__"
            + self.type_tag
            + self.no_shear_tag
            + self.align_light_dark_centre_tag
            + self.align_bulge_dark_centre_tag
            + self.fix_lens_light_tag
        )

    @property
    def fix_lens_light_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the setup folder as follows:

        fix_lens_light = False -> setup__
        fix_lens_light = True -> setup___fix_lens_light
        """
        if not self.fix_lens_light:
            return ""
        elif self.fix_lens_light:
            return "__fix_lens_light"
