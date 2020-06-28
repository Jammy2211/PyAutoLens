from autoconf import conf
from autogalaxy.pipeline import setup
from autolens import exc


class PipelineSetup(setup.PipelineSetup):
    def __init__(
        self,
        folders=None,
        hyper_galaxies=False,
        hyper_image_sky=False,
        hyper_background_noise=False,
        hyper_galaxies_search=None,
        inversion_search=None,
        hyper_combined_search=None,
        pixelization=None,
        regularization=None,
        lens_light_centre=None,
        align_light_mass_centre=False,
        align_bulge_disk_centre=False,
        align_bulge_disk_elliptical_comps=False,
        disk_as_sersic=False,
        number_of_gaussians=None,
        no_shear=False,
        lens_mass_centre=None,
        align_light_dark_centre=False,
        align_bulge_dark_centre=False,
        subhalo_instance=None,
        inversion_evidence_tolerance=None,
    ):
        """The setup of a pipeline, which controls how PyAutoLens template pipelines runs, for example controlling
        assumptions about the lens's light profile bulge-disk model or if shear is included in the mass model.

        Users can write their own pipelines which do not use or require the *PipelineSetup* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        pixelization : ag.pix.Pixelization
           If the pipeline uses an *Inversion* to reconstruct the galaxy's light, this determines the
           *Pixelization* used.
        regularization : ag.reg.Regularization
           If the pipeline uses an *Inversion* to reconstruct the galaxy's light, this determines the
           *Regularization* scheme used.
        lens_light_centre : (float, float)
           If input, a fixed (y,x) centre of the lens galaxy is used for the light profile model which is not treated 
           as a free parameter by the non-linear search.
        align_bulge_disk_centre : bool
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will align the centre of the bulge and disk components and not fit them separately.
        align_bulge_disk_phi : bool
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will align the rotation angles phi of the bulge and disk components and not fit them separately.
        align_bulge_disk_axis_ratio : bool
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will align the axis-ratios of the bulge and disk components and not fit them separately.
        disk_as_sersic : bool
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will use an EllipticalSersic for the disk instead of an EllipticalExponential.
        number_of_gaussians : int
            If a multi-Gaussian light model is used to fit the galaxy, this determines the number of Gaussians.
        no_shear : bool
            If True, shear is omitted from the mass model, if False it is included.
        lens_light_centre : (float, float)
           If input, a fixed (y,x) centre of the lens galaxy is used for the mass profile model which is not treated 
           as a free parameter by the non-linear search.
        align_light_mass_centre : bool
            If True, and the mass model is a decomposed single light and dark matter model (e.g. EllipticalSersic +
            SphericalNFW), the centre of the light and dark matter profiles are aligned.
        align_light_mass_centre : bool
            If True, and the mass model is a decomposed bulge, diskand dark matter model (e.g. EllipticalSersic +
            EllipticalExponential + SphericalNFW), the centre of the bulge and dark matter profiles are aligned.
        hyper_galaxies : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used to scale the
            noise-map of the dataset throughout the fitting.
        hyper_image_sky : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            image's background sky component in the model.
        hyper_background_noise : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            noise-map's background component in the model.
        """
        super().__init__(
            folders=folders,
            hyper_galaxies=hyper_galaxies,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            hyper_galaxies_search=hyper_galaxies_search,
            inversion_search=inversion_search,
            hyper_combined_search=hyper_combined_search,
            pixelization=pixelization,
            regularization=regularization,
            align_bulge_disk_centre=align_bulge_disk_centre,
            align_bulge_disk_elliptical_comps=align_bulge_disk_elliptical_comps,
            disk_as_sersic=disk_as_sersic,
            number_of_gaussians=number_of_gaussians,
            inversion_evidence_tolerance=inversion_evidence_tolerance,
        )

        self.no_shear = no_shear
        self.lens_light_centre = lens_light_centre
        self.lens_mass_centre = lens_mass_centre
        self.align_light_mass_centre = align_light_mass_centre

        if align_light_dark_centre and align_bulge_dark_centre:
            raise exc.SettingsException(
                "In PipelineMassSettings align_light_dark_centre and align_bulge_disk_centre"
                "can not both be True (one is not relevent to the light profile you are fitting"
            )

        self.align_light_dark_centre = align_light_dark_centre
        self.align_bulge_dark_centre = align_bulge_dark_centre

        self.subhalo_instance = subhalo_instance

    @property
    def tag(self):
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to.
        """
        return (
            conf.instance.tag.get("pipeline", "pipeline", str)
            + self.hyper_tag
            + self.inversion_tag
            + self.align_light_mass_centre_tag
            + self.lens_light_centre_tag
            + self.align_bulge_disk_tag
            + self.disk_as_sersic_tag
            + self.number_of_gaussians_tag
            + self.no_shear_tag
            + self.lens_mass_centre_tag
            + self.align_light_dark_centre_tag
            + self.align_bulge_dark_centre_tag
            + self.subhalo_centre_tag
            + self.subhalo_mass_at_200_tag
        )

    @property
    def no_shear_tag(self):
        """Generate a tag if an external shear is included in the mass model of the pipeline  are 
        fixedto a previous estimate, or varied during the analysis, to customize pipeline output paths..

        This changes the setup folder as follows:

        no_shear = False -> setup__with_shear
        no_shear = True -> setup___no_shear
        """
        if not self.no_shear:
            return "__" + conf.instance.tag.get("pipeline", "with_shear", str)
        elif self.no_shear:
            return "__" + conf.instance.tag.get("pipeline", "no_shear", str)

    @property
    def lens_light_centre_tag(self):
        """Generate a tag if the lens light model centre of the pipeline is fixed to an input value, to customize 
        pipeline output paths.

        This changes the setup folder as follows:

        lens_light_centre = None -> setup
        lens_light_centre = (1.0, 1.0) -> setup___lens_light_centre_(1.0, 1.0)
        lens_light_centre = (3.0, -2.0) -> setup___lens_light_centre_(3.0, -2.0)
        """
        if self.lens_light_centre is None:
            return ""
        else:
            y = "{0:.2f}".format(self.lens_light_centre[0])
            x = "{0:.2f}".format(self.lens_light_centre[1])
            return (
                "__"
                + conf.instance.tag.get("pipeline", "lens_light_centre", str)
                + "_("
                + y
                + ","
                + x
                + ")"
            )

    @property
    def lens_mass_centre_tag(self):
        """Generate a tag if the lens mass model centre of the pipeline is fixed to an input value, to customize 
        pipeline output paths.

        This changes the setup folder as follows:

        lens_mass_centre = None -> setup
        lens_mass_centre = (1.0, 1.0) -> setup___lens_mass_centre_(1.0, 1.0)
        lens_mass_centre = (3.0, -2.0) -> setup___lens_mass_centre_(3.0, -2.0)
        """
        if self.lens_mass_centre is None:
            return ""
        else:
            y = "{0:.2f}".format(self.lens_mass_centre[0])
            x = "{0:.2f}".format(self.lens_mass_centre[1])
            return (
                "__"
                + conf.instance.tag.get("pipeline", "lens_mass_centre", str)
                + "_("
                + y
                + ","
                + x
                + ")"
            )

    @property
    def align_light_mass_centre_tag(self):
        """Generate a tag if the lens mass model is centre is aligned with that of its light profile.

        This changes the setup folder as follows:

        align_light_mass_centre = False -> setup
        align_light_mass_centre = True -> setup___align_light_mass_centre
        """
        if self.lens_light_centre is not None and self.lens_mass_centre is not None:
            return ""

        if not self.align_light_mass_centre:
            return ""
        elif self.align_light_mass_centre:
            return "__" + conf.instance.tag.get(
                "pipeline", "align_light_mass_centre", str
            )

    @property
    def align_light_dark_centre_tag(self):
        """Generate a tag if the lens mass model is a decomposed light + dark matter model if their centres are aligned.

        This changes the setup folder as follows:

        align_light_dark_centre = False -> setup
        align_light_dark_centre = True -> setup___align_light_dark_centre
        """
        if not self.align_light_dark_centre:
            return ""
        elif self.align_light_dark_centre:
            return "__" + conf.instance.tag.get(
                "pipeline", "align_light_dark_centre", str
            )

    @property
    def align_bulge_dark_centre_tag(self):
        """Generate a tag if the lens mass model is a decomposed bulge + disk + dark matter model if the bulge centre
        is aligned with the dark matter centre.

        This changes the setup folder as follows:

        align_bulge_dark_centre = False -> setup
        align_bulge_dark_centre = True -> setup___align_bulge_dark_centre
        """
        if not self.align_bulge_dark_centre:
            return ""
        elif self.align_bulge_dark_centre:
            return "__" + conf.instance.tag.get(
                "pipeline", "align_bulge_dark_centre", str
            )

    @property
    def subhalo_centre_tag(self):
        """Generate a tag if the subhalo mass model centre of the pipeline is fixed to an input value, to customize 
        pipeline output paths.

        This changes the setup folder as follows:

        subhalo_centre = None -> setup
        subhalo_centre = (1.0, 1.0) -> setup___sub_centre_(1.0, 1.0)
        subhalo_centre = (3.0, -2.0) -> setup___sub_centre_(3.0, -2.0)
        """
        if self.subhalo_instance is None:
            return ""
        else:
            y = "{0:.2f}".format(self.subhalo_instance.centre[0])
            x = "{0:.2f}".format(self.subhalo_instance.centre[1])
            return (
                "__"
                + conf.instance.tag.get("pipeline", "subhalo_centre", str)
                + "_("
                + y
                + ","
                + x
                + ")"
            )

    @property
    def subhalo_mass_at_200_tag(self):
        """Generate a tag if the subhalo mass model mass_at_200 of the pipeline is fixed to an input value, to
        customize pipeline output paths.

        This changes the setup folder as follows:

        subhalo_mass_at_200 = None -> setup
        subhalo_mass_at_200 = 1e8 -> setup___sub_mass_1.0e+08
        subhalo_mass_at_200 = 1e9 -> setup___sub_mass_1.0e+09
        """
        if self.subhalo_instance is None:
            return ""
        else:

            return (
                "__"
                + conf.instance.tag.get("pipeline", "subhalo_mass_at_200", str)
                + "_"
                + "{0:.1e}".format(self.subhalo_instance.mass_at_200_input)
            )
