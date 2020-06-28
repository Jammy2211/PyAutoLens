import autofit as af
import autogalaxy as ag
from autoconf import conf
from autolens.pipeline import setup


class SLaM:
    def __init__(self, folders=None, hyper=None, source=None, light=None, mass=None):

        self.folders = folders
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

    @property
    def source_pipeline_tag(self):
        return conf.instance.tag.get("pipeline", "pipeline", str) + self.hyper.hyper_tag

    @property
    def lens_light_tag_for_source_pipeline(self):
        """Return the lens light tag in a *Source* pipeline, where the lens light is set depending on the source
        initialize pipeline that was used."""

        if self.light.type_tag is None:
            return ""

        light_tag = conf.instance.tag.get("pipeline", "light", str)
        return "__" + light_tag + "_" + self.light.type_tag

    def lens_with_previous_light_and_model_mass(self):
        """Setup the lens galaxy model using the previous results of a pipeline or phase.

        This function is required when linking a lens light model to a pipeline where the lens light is not specified
        in the pipeline itself. For example, when linking the source initialize parametric pipeline (which can use
         multiple differenet lens light models) to a source inversion pipeline."""

        lens = af.last[-1].instance.galaxies.lens
        lens.mass = af.last[-1].model.galaxies.lens.mass
        lens.shear = af.last[-1].model.galaxies.lens.shear

        lens.hyper_galaxy = (
            af.last.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy
        )

        return lens

    def source_from_previous_pipeline(self):
        """Setup the source model using the previous pipeline and phase results.

        The source light model is not specified by the pipeline light and mass pipelines (e.g. the previous pipelines
        are used to determine whether the source model is parametric or an inversion.

        The source is returned as a model if it is parametric (given it must be updated to properly compute a new mass
        model) whereas inversions are returned as an instance (as they have sufficient flexibility to typically not
        required updating). This behaviour can be customizzed in SLaM pipelines by replacing this method with the
        *source_from_previous_pipeline_model_or_instance* method of the SLaM class.

        The pipeline tool af.last is required to locate the previous source model, which requires an index based on the
        pipelines that have run. For example, if the source model you wish to load from is 3 phases back (perhaps
        because there were multiple phases in a Light pipeline preivously) this index should be 2.

        Parameters
        ----------
        source_as_model : bool
            If *True* the source is returned as a *model* where the parameters are fitted for using priors of the
            phase result it is loaded from. If *False*, it is an instance of that phase's result.
        index : integer
            The index (counting backwards from this phase) of the phase result used to setup the source.
        """
        if self.source.type_tag in "sersic":
            return self.source_from_previous_pipeline_model_or_instance(
                source_as_model=True, index=0
            )
        else:
            return self.source_from_previous_pipeline_model_or_instance(
                source_as_model=False, index=0
            )

    def source_from_previous_pipeline_model_or_instance(
        self, source_as_model=False, index=0
    ):
        """Setup the source model using the previous pipeline and phase results.

        The source light model is not specified by the pipeline light and mass pipelines (e.g. the previous pipelines
        are used to determine whether the source model is parametric or an inversion).

        The source can be returned as an instance or a model, depending on the optional input. The default SLaM p
        ipelines return parametric sources as a model (give they must be updated to properly compute a new mass model)
        and return inversions as an instance (as they have sufficient flexibility to typically not required updating).
        They use the *source_from_pevious_pipeline* method of the SLaM class to do this.

        The pipeline tool af.last is required to locate the previous source model, which requires an index based on the
        pipelines that have run. For example, if the source model you wish to load from is 3 phases back (perhaps
        because there were multiple phases in a Light pipeline preivously) this index should be 2.

        Parameters
        ----------
        source_as_model : bool
            If *True* the source is returned as a *model* where the parameters are fitted for using priors of the
            phase result it is loaded from. If *False*, it is an instance of that phase's result.
        index : integer
            The index (counting backwards from this phase) of the phase result used to setup the source.
        """

        if self.hyper.hyper_galaxies:

            hyper_galaxy = af.PriorModel(ag.HyperGalaxy)

            hyper_galaxy.noise_factor = (
                af.last.hyper_combined.model.galaxies.source.hyper_galaxy.noise_factor
            )
            hyper_galaxy.contribution_factor = (
                af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.contribution_factor
            )
            hyper_galaxy.noise_power = (
                af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.noise_power
            )

        else:

            hyper_galaxy = None

        if self.source.type_tag in "sersic":

            if source_as_model:

                return ag.GalaxyModel(
                    redshift=af.last[index].model.galaxies.source.redshift,
                    sersic=af.last[index].model.galaxies.source.sersic,
                    hyper_galaxy=hyper_galaxy,
                )

            else:

                return ag.GalaxyModel(
                    redshift=af.last[index].instance.galaxies.source.redshift,
                    sersic=af.last[index].instance.galaxies.source.sersic,
                    hyper_galaxy=hyper_galaxy,
                )

        else:

            if source_as_model:

                return ag.GalaxyModel(
                    redshift=af.last.instance.galaxies.source.redshift,
                    pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
                    regularization=af.last.hyper_combined.model.galaxies.source.regularization,
                )

            else:

                return ag.GalaxyModel(
                    redshift=af.last.instance.galaxies.source.redshift,
                    pixelization=af.last[
                        index
                    ].hyper_combined.instance.galaxies.source.pixelization,
                    regularization=af.last[
                        index
                    ].hyper_combined.instance.galaxies.source.regularization,
                    hyper_galaxy=hyper_galaxy,
                )

    def lens_from_previous_pipeline(self, redshift_lens, mass, shear):
        """For the SLaM mass pipeline, return the lens _GalaxyModel_, where:

        1) The lens light model uses the light model of the Light pipeline.
        2) The lens light is returned as a model if *fix_lens_light* is *False, an instance if *True*.

        Parameters
        ----------
        redshift_lens : float
            The redshift of the lens galaxy.
        mass : ag.MassProfile
            The mass model of the len galaxy.
        shear : ag.ExternalShear
            The external shear of the lens galaxy.
        """

        if self.mass.fix_lens_light:

            return ag.GalaxyModel(
                redshift=redshift_lens,
                bulge=af.last.instance.galaxies.lens.bulge,
                disk=af.last.instance.galaxies.lens.disk,
                mass=mass,
                shear=shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            )

        else:

            return ag.GalaxyModel(
                redshift=redshift_lens,
                bulge=af.last.model.galaxies.lens.bulge,
                disk=af.last.model.galaxies.lens.disk,
                mass=mass,
                shear=shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            )


class HyperSetup(setup.PipelineSetup):
    def __init__(
        self,
        hyper_galaxies=False,
        hyper_image_sky=False,
        hyper_background_noise=False,
        hyper_fixed_after_source=False,
        hyper_galaxies_search=None,
        inversion_search=None,
        hyper_combined_search=None,
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
            hyper_galaxies_search=hyper_galaxies_search,
            inversion_search=inversion_search,
            hyper_combined_search=hyper_combined_search,
        )

        self.hyper_fixed_after_source = hyper_fixed_after_source

    @property
    def tag(self):
        return (
            conf.instance.tag.get("pipeline", "pipeline", str)
            + self.hyper_tag
            + self.hyper_fixed_after_source_tag
        )

    @property
    def hyper_tag(self):
        """Tag ithe hyper pipeline features used in a hyper pipeline to customize pipeline output paths.
        """
        if not any(
            [self.hyper_galaxies, self.hyper_image_sky, self.hyper_background_noise]
        ):
            return ""

        return (
            "__"
            + conf.instance.tag.get("pipeline", "hyper", str)
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
            return "_" + conf.instance.tag.get(
                "pipeline", "hyper_fixed_after_source", str
            )


class SourceSetup(setup.PipelineSetup):
    def __init__(
        self,
        pixelization=None,
        regularization=None,
        no_shear=False,
        lens_light_centre=None,
        lens_mass_centre=None,
        align_light_mass_centre=False,
        lens_light_bulge_only=False,
        number_of_gaussians=None,
        inversion_evidence_tolerance=None,
    ):

        super().__init__(
            pixelization=pixelization,
            regularization=regularization,
            no_shear=no_shear,
            lens_light_centre=lens_light_centre,
            lens_mass_centre=lens_mass_centre,
            number_of_gaussians=number_of_gaussians,
            align_light_mass_centre=align_light_mass_centre,
            inversion_evidence_tolerance=inversion_evidence_tolerance,
        )

        self.lens_light_bulge_only = lens_light_bulge_only
        self.type_tag = None

    @property
    def tag(self):
        return (
            conf.instance.tag.get("pipeline", "source", str)
            + "__"
            + self.type_tag
            + self.number_of_gaussians_tag
            + self.no_shear_tag
            + self.lens_light_centre_tag
            + self.lens_mass_centre_tag
            + self.align_light_mass_centre_tag
            + self.lens_light_bulge_only_tag
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
            return "__" + conf.instance.tag.get("pipeline", "bulge_only", str)

    @property
    def shear(self):
        """For a SLaM source pipeline, determine the shear model from the no_shear setting."""
        if not self.no_shear:
            return ag.mp.ExternalShear

    @property
    def is_inversion(self):
        """Returns True is the source is an inversion, meaning this SLaM analysis has gone through a source
        inversion pipeline."""
        if self.type_tag in "sersic":
            return False
        else:
            return True

    def align_centre_of_mass_to_light(self, mass, light_centre):
        """Align the centre of a mass profile to the centre of a light profile, if the align_light_mass_centre
        SLaM setting is True.
        
        Parameters
        ----------
        mass : ag.mp.MassProfile
            The mass profile whose centre may be aligned with the lens_light_centre attribute.
        light : (float, float)
            The centre of the light profile the mass profile is aligned with.
        """
        if self.align_light_mass_centre:
            mass.centre = light_centre
        else:
            mass.centre.centre_0 = af.GaussianPrior(mean=light_centre[0], sigma=0.1)
            mass.centre.centre_1 = af.GaussianPrior(mean=light_centre[1], sigma=0.1)
        return mass

    def align_centre_to_lens_light_centre(self, light):
        """
        Align the centre of an input light profile to the lens_light_centre of this instance of the SLaM Source
        class, make the centre of the light profile fixed and thus not free parameters that are fitted for.

        If the lens_light_centre is not input (and thus None) the light profile centre is unchanged.

        Parameters
        ----------
        light : ag.mp.MassProfile
            The light profile whose centre may be aligned with the lens_light_centre attribute.
        """
        if self.lens_light_centre is not None:
            light.centre = self.lens_light_centre
        return light

    def align_centre_to_lens_mass_centre(self, mass):
        """
        Align the centre of an input mass profile to the lens_mass_centre of this instance of the SLaM Source
        class, make the centre of the mass profile fixed and thus not free parameters that are fitted for.

        If the lens_mass_centre is not input (and thus None) the mass profile centre is unchanged.

        Parameters
        ----------
        mass : ag.mp.MassProfile
            The mass profile whose centre may be aligned with the lens_mass_centre attribute.
        """
        if self.lens_mass_centre is not None:
            mass.centre = self.lens_mass_centre
        return mass

    def remove_disk_from_lens_galaxy(self, lens):
        """Remove the disk from a GalaxyModel, if the SLaM settings specify to mooel the lens's light as bulge-only."""
        if self.lens_light_bulge_only:
            lens.disk = None
        return lens

    def unfix_lens_mass_centre(self, mass):
        """If the centre of a mass model was previously fixed to an input value (e.g. lens_mass_centre), unaligned it
        by making its centre GaussianPriors.
        """

        if self.lens_mass_centre is not None:

            mass.centre.centre_0 = af.GaussianPrior(
                mean=self.lens_mass_centre[0], sigma=0.05
            )
            mass.centre.centre_1 = af.GaussianPrior(
                mean=self.lens_mass_centre[1], sigma=0.05
            )

        return mass

    def unalign_lens_mass_centre_from_light_centre(self, mass):
        """If the centre of a mass model was previously aligned with that of the lens light centre, unaligned them
        by using an earlier model of the light.
        """
        if self.align_light_mass_centre:

            mass.centre = af.last[-3].model.galaxies.lens.bulge.centre

        return mass


class LightSetup(setup.PipelineSetup):
    def __init__(
        self,
        align_bulge_disk_centre=False,
        align_bulge_disk_elliptical_comps=False,
        disk_as_sersic=False,
        number_of_gaussians=None,
    ):

        super().__init__(
            align_bulge_disk_centre=align_bulge_disk_centre,
            align_bulge_disk_elliptical_comps=align_bulge_disk_elliptical_comps,
            disk_as_sersic=disk_as_sersic,
            number_of_gaussians=number_of_gaussians,
        )

        self.type_tag = None

    @property
    def tag(self):
        if self.number_of_gaussians is None:
            return (
                conf.instance.tag.get("pipeline", "light", str)
                + "__"
                + self.type_tag
                + self.align_bulge_disk_tag
                + self.disk_as_sersic_tag
            )
        else:
            return (
                conf.instance.tag.get("pipeline", "light", str)
                + "__"
                + self.type_tag
                + self.number_of_gaussians_tag
            )


class MassSetup(setup.PipelineSetup):
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
            conf.instance.tag.get("pipeline", "mass", str)
            + "__"
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
            return "__" + conf.instance.tag.get("pipeline", "fix_lens_light", str)

    @property
    def shear_from_previous_pipeline(self):
        """Return the shear PriorModel from a previous pipeline, where:

        1) If the shear was included in the *Source* pipeline and *no_shear* is *False* in the *Mass* object, it is
           returned using this pipeline result as a model.
        2) If the shear was not included in the *Source* pipeline and *no_shear* is *False* in the *Mass* object, it is
            returned as a new *ExternalShear* PriorModel.
        3) If *no_shear* is *True* in the *Mass* object, it is returned as None and omitted from the lens model.
        """
        if not self.no_shear:
            if af.last.model.galaxies.lens.shear is not None:
                return af.last.model.galaxies.lens.shear
            else:
                return ag.mp.ExternalShear
        else:
            return None
