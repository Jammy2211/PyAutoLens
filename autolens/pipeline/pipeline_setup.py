import autofit as af
import autoastro as aast
from autolens import exc
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg


class Setup(object):
    def __init__(self, general=None, source=None, light=None, mass=None):

        self.general = general
        self.source = source
        self.light = light
        self.mass = mass


class General(object):
    def __init__(
        self, hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
    ):

        self.hyper_galaxies = hyper_galaxies
        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

    @property
    def tag(self):
        return "general" + self.hyper_tag

    @property
    def hyper_tag(self):

        if not any(
            [self.hyper_galaxies, self.hyper_image_sky, self.hyper_background_noise]
        ):
            return ""

        return (
            "__hyper"
            + self.hyper_galaxies_tag
            + self.hyper_image_sky_tag
            + self.hyper_background_noise_tag
        )

    @property
    def hyper_galaxies_tag(self):
        """Generate a tag for if hyper-galaxies are used in a hyper_galaxies pipeline to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___hyper_galaxies
        """
        if not self.hyper_galaxies:
            return ""
        elif self.hyper_galaxies:
            return "_galaxies"

    @property
    def hyper_image_sky_tag(self):
        """Generate a tag for if the sky-background is hyper as a hyper_galaxies-parameter in a hyper_galaxies pipeline to
        customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___hyper_bg_sky
        """
        if not self.hyper_image_sky:
            return ""
        elif self.hyper_image_sky:
            return "_bg_sky"

    @property
    def hyper_background_noise_tag(self):
        """Generate a tag for if the background noise is hyper as a hyper_galaxies-parameter in a hyper_galaxies pipeline to
        customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___hyper_bg_noise
        """
        if not self.hyper_background_noise:
            return ""
        elif self.hyper_background_noise:
            return "_bg_noise"


class Source(object):
    def __init__(
        self,
        pixelization=pix.VoronoiBrightnessImage,
        regularization=reg.AdaptiveBrightness,
        no_shear=False,
        lens_light_centre=None,
        lens_mass_centre=None,
        align_light_mass_centre=False,
        lens_light_bulge_only=False,
        fix_lens_light=False,
    ):

        self.pixelization = pixelization
        self.regularization = regularization
        self.no_shear = no_shear
        self.lens_light_centre = lens_light_centre
        self.lens_mass_centre = lens_mass_centre
        self.align_light_mass_centre = align_light_mass_centre
        self.lens_light_bulge_only = lens_light_bulge_only
        self.fix_lens_light = fix_lens_light

    @property
    def tag(self):
        return (
            "source"
            + self.inversion_tag
            + self.no_shear_tag
            + self.lens_light_centre_tag
            + self.lens_mass_centre_tag
            + self.align_light_mass_centre_tag
            + self.lens_light_bulge_only_tag
            + self.fix_lens_light_tag
        )

    @property
    def tag_no_inversion(self):
        return (
            "source"
            + self.no_shear_tag
            + self.lens_light_centre_tag
            + self.lens_mass_centre_tag
            + self.align_light_mass_centre_tag
            + self.lens_light_bulge_only_tag
            + self.fix_lens_light_tag
        )

    @property
    def tag_beginner(self):
        return "source" + self.inversion_tag

    @property
    def tag_beginner_no_inversion(self):
        return "source"

    def tag_from_source(self, source):

        source_is_inversion = source_is_inversion_from_source(source=source)

        if source_is_inversion:
            source_tag = self.inversion_tag
        else:
            source_tag = "__parametric"

        return (
            "source"
            + source_tag
            + self.no_shear_tag
            + self.lens_light_centre_tag
            + self.lens_mass_centre_tag
            + self.align_light_mass_centre_tag
            + self.lens_light_bulge_only_tag
            + self.fix_lens_light_tag
        )

    @property
    def inversion_tag(self):
        return self.pixelization_tag + self.regularization_tag

    @property
    def pixelization_tag(self):

        if self.pixelization is None:
            return ""
        else:
            return "__pix_" + af.conf.instance.label.get(
                "tag", self.pixelization().__class__.__name__, str
            )

    @property
    def regularization_tag(self):

        if self.regularization is None:
            return ""
        else:
            return "__reg_" + af.conf.instance.label.get(
                "tag", self.regularization().__class__.__name__, str
            )

    @property
    def no_shear_tag(self):
        """Generate a tag for if an external shear is included in the mass model of the pipeline and / or phase are fixed
        to a previous estimate, or varied during he analysis, to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___with_shear
        """
        if not self.no_shear:
            return "__with_shear"
        elif self.no_shear:
            return "__no_shear"

    @property
    def lens_light_centre_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___fix_lens_light
        """
        if self.lens_light_centre is None:
            return ""
        else:
            y = "{0:.2f}".format(self.lens_light_centre[0])
            x = "{0:.2f}".format(self.lens_light_centre[1])
            return "__lens_light_centre_(" + y + "," + x + ")"

    @property
    def lens_mass_centre_tag(self):
        """Generate a tag for if the lens mass of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_mass = False -> pipeline_name__
        fix_lens_mass = True -> pipeline_name___fix_lens_mass
        """
        if self.lens_mass_centre is None:
            return ""
        else:
            y = "{0:.2f}".format(self.lens_mass_centre[0])
            x = "{0:.2f}".format(self.lens_mass_centre[1])
            return "__lens_mass_centre_(" + y + "," + x + ")"

    @property
    def align_light_mass_centre_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___fix_lens_light
        """
        if self.lens_light_centre is not None and self.lens_mass_centre is not None:
            return ""

        if not self.align_light_mass_centre:
            return ""
        elif self.align_light_mass_centre:
            return "__align_light_mass_centre"

    @property
    def lens_light_bulge_only_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___fix_lens_light
        """
        if not self.lens_light_bulge_only:
            return ""
        elif self.lens_light_bulge_only:
            return "__bulge_only"

    @property
    def fix_lens_light_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___fix_lens_light
        """
        if not self.fix_lens_light:
            return ""
        elif self.fix_lens_light:
            return "__fix_lens_light"


class Light(object):
    def __init__(
        self,
        align_bulge_disk_centre=False,
        align_bulge_disk_phi=False,
        align_bulge_disk_axis_ratio=False,
        disk_as_sersic=False,
    ):

        self.align_bulge_disk_centre = align_bulge_disk_centre
        self.align_bulge_disk_phi = align_bulge_disk_phi
        self.align_bulge_disk_axis_ratio = align_bulge_disk_axis_ratio
        self.disk_as_sersic = disk_as_sersic

    @property
    def tag(self):
        return "light" + self.align_bulge_disk_tag + self.disk_as_sersic_tag

    def tag_from_lens(self, lens):

        lens_light_tag = lens_light_tag_from_lens(lens=lens)

        return (
            "light"
            + lens_light_tag
            + self.align_bulge_disk_tag
            + self.disk_as_sersic_tag
        )

    @property
    def align_bulge_disk_centre_tag(self):
        """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
        based on the bulge-disk model. This changee the phase name 'pipeline_name__' as follows:

        bd_align_centres = False -> pipeline_name__
        bd_align_centres = True -> pipeline_name___bd_align_centres
        """
        if not self.align_bulge_disk_centre:
            return ""
        elif self.align_bulge_disk_centre:
            return "_centre"

    @property
    def align_bulge_disk_axis_ratio_tag(self):
        """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
        based on the bulge-disk model. This changes the phase name 'pipeline_name__' as follows:

        bd_align_axis_ratio = False -> pipeline_name__
        bd_align_axis_ratio = True -> pipeline_name___bd_align_axis_ratio
        """
        if not self.align_bulge_disk_axis_ratio:
            return ""
        elif self.align_bulge_disk_axis_ratio:
            return "_axis_ratio"

    @property
    def align_bulge_disk_phi_tag(self):
        """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
        based on the bulge-disk model. This changes the phase name 'pipeline_name__' as follows:

        bd_align_phi = False -> pipeline_name__
        bd_align_phi = True -> pipeline_name___bd_align_phi
        """
        if not self.align_bulge_disk_phi:
            return ""
        elif self.align_bulge_disk_phi:
            return "_phi"

    @property
    def align_bulge_disk_tag(self):
        """Generate a tag for the alignment of the geometry of the bulge and disk of a bulge-disk system, to customize \
        phase names based on the bulge-disk model. This adds together the bulge_disk tags generated in the 3 functions
        above
        """

        if not any(
            [
                self.align_bulge_disk_centre,
                self.align_bulge_disk_axis_ratio,
                self.align_bulge_disk_phi,
            ]
        ):
            return ""

        return (
            "__align_bulge_disk"
            + self.align_bulge_disk_centre_tag
            + self.align_bulge_disk_axis_ratio_tag
            + self.align_bulge_disk_phi_tag
        )

    @property
    def disk_as_sersic_tag(self):
        """Generate a tag for if the disk component of a bulge-disk light profile fit of the pipeline is modeled as a \
        Sersic or the default profile of an Exponential.

        This changes the phase name 'pipeline_name__' as follows:

        disk_as_sersic = False -> pipeline_name__
        disk_as_sersic = True -> pipeline_name___disk_as_sersic
        """
        if not self.disk_as_sersic:
            return "__disk_exp"
        elif self.disk_as_sersic:
            return "__disk_sersic"


class Mass(object):
    def __init__(
        self,
        no_shear=False,
        align_light_dark_centre=False,
        align_bulge_dark_centre=False,
        fix_lens_light=False,
    ):

        self.no_shear = no_shear

        if align_light_dark_centre and align_bulge_dark_centre:
            raise exc.SettingsException(
                "In PipelineMassSettings align_light_dark_centre and align_bulge_disk_centre"
                "can not both be True (one is not relevent to the light profile you are fitting"
            )

        self.align_light_dark_centre = align_light_dark_centre
        self.align_bulge_dark_centre = align_bulge_dark_centre

        self.fix_lens_light = fix_lens_light

    @property
    def tag(self):
        return (
            "mass"
            + self.no_shear_tag
            + self.align_light_dark_centre_tag
            + self.align_bulge_dark_centre_tag
            + self.fix_lens_light_tag
        )

    @property
    def no_shear_tag(self):
        """Generate a tag for if an external shear is included in the mass model of the pipeline and / or phase are fixed
        to a previous estimate, or varied during he analysis, to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___with_shear
        """
        if not self.no_shear:
            return "__with_shear"
        elif self.no_shear:
            return "__no_shear"

    @property
    def align_light_dark_centre_tag(self):
        """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
        based on the bulge-disk model. This changee the phase name 'pipeline_name__' as follows:

        bd_align_centres = False -> pipeline_name__
        bd_align_centres = True -> pipeline_name___bd_align_centres
        """
        if not self.align_light_dark_centre:
            return ""
        elif self.align_light_dark_centre:
            return "__align_light_dark_centre"

    @property
    def align_bulge_dark_centre_tag(self):
        """Generate a tag for if the bulge and dark of a bulge-dark system are aligned or not, to customize phase names \
        based on the bulge-dark model. This changee the phase name 'pipeline_name__' as follows:

        bd_align_centres = False -> pipeline_name__
        bd_align_centres = True -> pipeline_name___bd_align_centres
        """
        if not self.align_bulge_dark_centre:
            return ""
        elif self.align_bulge_dark_centre:
            return "__align_bulge_dark_centre"

    @property
    def fix_lens_light_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___fix_lens_light
        """
        if not self.fix_lens_light:
            return ""
        elif self.fix_lens_light:
            return "__fix_lens_light"


def lens_light_tag_from_lens(lens):

    if hasattr(lens, "sersic") or hasattr(lens, "light"):
        return "__bulge_disk"
    elif hasattr(lens, "bulge") or hasattr(lens, "disk"):
        return "__bulge_disk"
    else:
        return ""


def lens_from_result(result, fix_lens_light):

    if hasattr(result, "light"):

        if fix_lens_light:

            light = result.instance.galaxies.lens.light

        else:

            light = result.model.galaxies.lens.light

        return aast.GalaxyModel(
            redshift=result.instance.galaxies.lens.redshift, light=light
        )

    elif hasattr(result, "sersic"):

        if fix_lens_light:

            sersic = result.instance.galaxies.lens.sersic

        else:

            sersic = result.model.galaxies.lens.sersic

        return aast.GalaxyModel(
            redshift=result.instance.galaxies.lens.redshift, sersic=sersic
        )

    elif hasattr(result, "bulge"):

        if fix_lens_light:

            bulge = result.instance.galaxies.lens.bulge
            disk = result.instance.galaxies.lens.disk

        else:

            bulge = result.model.galaxies.lens.bulge
            disk = result.model.galaxies.lens.disk

        return aast.GalaxyModel(
            redshift=result.instance.galaxies.lens.redshift, bulge=bulge, disk=disk
        )


def source_is_inversion_from_source(source):

    if source.pixelization is None:

        return False

    else:

        return True


def source_from_result(result, include_hyper_source):

    if include_hyper_source:
        hyper_galaxy = (
            af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy
        )
        hyper_galaxy.noise_factor = (
            af.last.hyper_combined.model.galaxies.source.hyper_galaxy.noise_factor
        )
    else:
        hyper_galaxy = None

    if result.model.galaxies.source.pixelization is None:

        return aast.GalaxyModel(
            redshift=result.instance.galaxies.source.redshift,
            light=result.model.galaxies.source.light,
            hyper_galaxy=hyper_galaxy,
        )

    else:

        return aast.GalaxyModel(
            redshift=result.instance.galaxies.source.redshift,
            pixelization=result.instance.galaxies.source.pixelization,
            regularization=result.instance.galaxies.source.regularization,
            hyper_galaxy=hyper_galaxy,
        )
