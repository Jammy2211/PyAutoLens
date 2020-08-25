import autofit as af
from autoconf import conf
from autogalaxy.pipeline import setup
from autogalaxy.profiles import mass_profiles as mp
from autogalaxy.profiles import light_and_mass_profiles as lmp
from autolens import exc


class SetupPipeline(setup.SetupPipeline):
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
        constant_mass_to_light_ratio=False,
        bulge_mass_to_light_ratio_gradient=False,
        disk_mass_to_light_ratio_gradient=False,
        align_light_dark_centre=False,
        align_bulge_dark_centre=False,
        include_smbh=False,
        smbh_centre_fixed=True,
        subhalo_instance=None,
        inversion_pixels_fixed=None,
        evidence_tolerance=None,
    ):
        """The setup of a pipeline, which controls how PyAutoLens template pipelines runs, for example controlling
        assumptions about the lens's light profile bulge-disk model or if shear is included in the mass model.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

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
            inversion_pixels_fixed=inversion_pixels_fixed,
            evidence_tolerance=evidence_tolerance,
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

        self.constant_mass_to_light_ratio = constant_mass_to_light_ratio
        self.bulge_mass_to_light_ratio_gradient = bulge_mass_to_light_ratio_gradient
        self.disk_mass_to_light_ratio_gradient = disk_mass_to_light_ratio_gradient
        self.align_light_dark_centre = align_light_dark_centre
        self.align_bulge_dark_centre = align_bulge_dark_centre
        self.include_smbh = include_smbh
        self.smbh_centre_fixed = smbh_centre_fixed

        self.subhalo_instance = subhalo_instance

    @property
    def tag(self):
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to.
        """
        return (
            conf.instance.tag.get("pipeline", "pipeline")
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
            + self.include_smbh_tag
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
            return "__" + conf.instance.tag.get("pipeline", "with_shear")
        return "__" + conf.instance.tag.get("pipeline", "no_shear")

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

        y = "{0:.2f}".format(self.lens_light_centre[0])
        x = "{0:.2f}".format(self.lens_light_centre[1])
        return (
            "__"
            + conf.instance.tag.get("pipeline", "lens_light_centre")
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

        y = "{0:.2f}".format(self.lens_mass_centre[0])
        x = "{0:.2f}".format(self.lens_mass_centre[1])
        return (
            "__"
            + conf.instance.tag.get("pipeline", "lens_mass_centre")
            + "_("
            + y
            + ","
            + x
            + ")"
        )

    @property
    def mass_to_light_ratio_tag(self):
        mass_to_light_tag = f"__{conf.instance.tag.get('pipeline', 'mass_to_light_ratio')}{self.constant_mass_to_light_ratio_tag}"

        if (
            self.bulge_mass_to_light_ratio_gradient
            or self.disk_mass_to_light_ratio_gradient
        ):
            gradient_tag = conf.instance.tag.get(
                "pipeline", "mass_to_light_ratio_gradient"
            )
            if self.bulge_mass_to_light_ratio_gradient:
                gradient_tag = (
                    f"{gradient_tag}{self.bulge_mass_to_light_ratio_gradient_tag}"
                )
            if self.disk_mass_to_light_ratio_gradient:
                gradient_tag = (
                    f"{gradient_tag}{self.disk_mass_to_light_ratio_gradient_tag}"
                )
            return f"{mass_to_light_tag}_{gradient_tag}"
        else:
            return mass_to_light_tag

    @property
    def constant_mass_to_light_ratio_tag(self):
        """Generate a tag for whether the mass-to-light ratio in a light-dark mass model is constaant (shared amongst
         all light and mass profiles) or free (all mass-to-light ratios are free parameters).

        This changes the setup folder as follows:

        constant_mass_to_light_ratio = False -> mlr_free
        constant_mass_to_light_ratio = True -> mlr_constant
        """
        if self.constant_mass_to_light_ratio:
            return (
                f"_{conf.instance.tag.get('pipeline', 'constant_mass_to_light_ratio')}"
            )
        return f"_{conf.instance.tag.get('pipeline', 'free_mass_to_light_ratio')}"

    @property
    def bulge_mass_to_light_ratio_gradient_tag(self):
        """Generate a tag for whether the mass-to-light ratio in a light-dark mass model is constaant (shared amongst
         all light and mass profiles) or free (all mass-to-light ratios are free parameters).

        This changes the setup folder as follows:

        constant_mass_to_light_ratio = False -> mlr_free
        constant_mass_to_light_ratio = True -> mlr_constant
        """
        if not self.bulge_mass_to_light_ratio_gradient:
            return ""
        return f"_{conf.instance.tag.get('pipeline', 'bulge_mass_to_light_ratio_gradient')}"

    @property
    def disk_mass_to_light_ratio_gradient_tag(self):
        """Generate a tag for whether the mass-to-light ratio in a light-dark mass model is constaant (shared amongst
         all light and mass profiles) or free (all mass-to-light ratios are free parameters).

        This changes the setup folder as follows:

        constant_mass_to_light_ratio = False -> mlr_free
        constant_mass_to_light_ratio = True -> mlr_constant
        """
        if not self.disk_mass_to_light_ratio_gradient:
            return ""
        return (
            f"_{conf.instance.tag.get('pipeline', 'disk_mass_to_light_ratio_gradient')}"
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
        return "__" + conf.instance.tag.get("pipeline", "align_light_mass_centre")

    @property
    def align_light_dark_centre_tag(self):
        """Generate a tag if the lens mass model is a decomposed light + dark matter model if their centres are aligned.

        This changes the setup folder as follows:

        align_light_dark_centre = False -> setup
        align_light_dark_centre = True -> setup___align_light_dark_centre
        """
        if not self.align_light_dark_centre:
            return ""
        return f"__{conf.instance.tag.get('pipeline', 'align_light_dark_centre')}"

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
        return "__" + conf.instance.tag.get("pipeline", "align_bulge_dark_centre")

    @property
    def include_smbh_tag(self):
        """Generate a tag if the lens mass model includes a _PointMass_ representing a super-massive black hole (smbh).
        
        The tag includes whether the _PointMass_ centre is fixed or fitted for as a free parameter.

        This changes the setup folder as follows:

        include_smbh = False -> setup
        include_smbh = True, smbh_centre_fixed=True -> setup___smbh_centre_fixed
        include_smbh = True, smbh_centre_fixed=False -> setup___smbh_centre_free
        """
        if not self.include_smbh:
            return ""

        include_smbh_tag = conf.instance.tag.get("pipeline", "include_smbh")

        if self.smbh_centre_fixed:

            smbh_centre_tag = conf.instance.tag.get("pipeline", "smbh_centre_fixed")

        else:

            smbh_centre_tag = conf.instance.tag.get("pipeline", "smbh_centre_free")

        return f"__{include_smbh_tag}_{smbh_centre_tag}"

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
                + conf.instance.tag.get("pipeline", "subhalo_centre")
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
                + conf.instance.tag.get("pipeline", "subhalo_mass_at_200")
                + "_"
                + "{0:.1e}".format(self.subhalo_instance.mass_at_200_input)
            )

    @property
    def bulge_light_and_mass_profile(self):
        """
        The light and mass profile of a bulge component of a galaxy.

        By default, this is returned as an  _EllipticalSersic_ profile without a radial gradient, however
        the _SetupPipeline_ inputs can be customized to change this to include a radial gradient.
        """
        if not self.bulge_mass_to_light_ratio_gradient:
            return af.PriorModel(lmp.EllipticalSersic)
        return af.PriorModel(lmp.EllipticalSersicRadialGradient)

    @property
    def disk_light_and_mass_profile(self):
        """
        The light and mass profile of a disk component of a galaxy.

        By default, this is returned as an  _EllipticalExponential_ profile without a radial gradient, however
        the _SetupPipeline_ inputs can be customized to change this to an _EllipticalSersic_ or to include a radial
        gradient.
        """

        if self.disk_as_sersic:
            if not self.disk_mass_to_light_ratio_gradient:
                return af.PriorModel(lmp.EllipticalSersic)
            return af.PriorModel(lmp.EllipticalSersicRadialGradient)
        else:
            if not self.disk_mass_to_light_ratio_gradient:
                return af.PriorModel(lmp.EllipticalExponential)
            return af.PriorModel(lmp.EllipticalExponentialRadialGradient)

    def set_mass_to_light_ratios_of_light_and_mass_profiles(
        self, light_and_mass_profiles
    ):
        """
        For an input list of _LightMassProfile_'s which will represent a galaxy with a light-dark mass model, set all
        the mass-to-light ratios of every light and mass profile to the same value if a constant mass-to-light ratio
        is being used, else keep them as free parameters.

        Parameters
        ----------
        light_and_mass_profiles : [LightMassProfile]
            The light and mass profiles which have their mass-to-light ratios changed.
        """

        if self.constant_mass_to_light_ratio:

            for profile in light_and_mass_profiles[1:]:

                profile.mass_to_light_ratio = light_and_mass_profiles[
                    0
                ].mass_to_light_ratio

    def smbh_from_centre(self, centre, centre_sigma=0.1):
        """
        Create a _PriorModel_ of a _PointMass_ _MassProfile_ if *include_smbh* is True, which is fitted for in the
        mass-model too represent a super-massive black-hole (smbh).

        The centre of the smbh is an input parameter of the functiono, and this centre is either fixed to the input
        values as an instance or fitted for as a model.

        Parameters
        ----------
        centre : (float, float)
            The centre of the _PointMass_ that repreents the super-massive black hole.
        centre_fixed : bool
            If True, the centre is fixed to the input values, else it is fitted for as free parameters.
        centre_sigma : float
            If the centre is free, this is the sigma value of each centre's _GaussianPrior_.
        """
        if not self.include_smbh:
            return None

        smbh = af.PriorModel(mp.PointMass)

        if self.smbh_centre_fixed:
            smbh.centre = centre
        else:
            smbh.centre.centre_0 = af.GaussianPrior(mean=centre[0], sigma=centre_sigma)
            smbh.centre.centre_1 = af.GaussianPrior(mean=centre[1], sigma=centre_sigma)

        return smbh
