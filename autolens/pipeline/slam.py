import autofit as af
import autogalaxy as ag
from autoconf import conf
from autogalaxy.pipeline import setup as ag_setup
from autolens.pipeline import setup


class AbstractSLaMPipeline:
    def __init__(self, setup_light, setup_mass, setup_source):

        self.setup_light = setup_light
        self.setup_mass = setup_mass
        self.setup_source = setup_source

    @property
    def shear(self):
        """For a SLaM source pipeline, determine the shear model from the no_shear setting."""
        if not self.setup_mass.no_shear:
            return ag.mp.ExternalShear


class SLaMPipelineSourceParametric(AbstractSLaMPipeline):
    def __init__(
        self,
        setup_light: setup.SetupLightBulgeDisk = setup.SetupLightBulgeDisk(),
        setup_mass: setup.SetupMassTotal = setup.SetupMassTotal(),
        setup_source: ag_setup.SetupSourceSersic = ag_setup.SetupSourceSersic(),
    ):

        super().__init__(
            setup_light=setup_light, setup_mass=setup_mass, setup_source=setup_source
        )


class SLaMPipelineSourceInversion(AbstractSLaMPipeline):
    def __init__(
        self, setup_source: setup.SetupSourceInversion = setup.SetupSourceInversion()
    ):

        super().__init__(setup_light=None, setup_mass=None, setup_source=setup_source)


class SLaMPipelineLight(AbstractSLaMPipeline):
    def __init__(
        self,
        setup_light: setup.SetupLightBulgeDisk = setup.SetupLightBulgeDisk(),
        setup_mass: setup.SetupMassTotal = setup.SetupMassTotal(),
    ):

        super().__init__(
            setup_source=None, setup_light=setup_light, setup_mass=setup_mass
        )


class SLaMPipelineMass(AbstractSLaMPipeline):
    def __init__(
        self,
        setup_mass: setup.SetupMassTotal = setup.SetupMassTotal(),
        fix_lens_light=True,
    ):

        super().__init__(setup_source=None, setup_light=None, setup_mass=setup_mass)

        self.fix_lens_light = fix_lens_light

    @property
    def fix_lens_light_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate,
        or varied during he analysis, to customize phase names.

        This changes the setup folder as follows:

        fix_lens_light = False -> setup__
        fix_lens_light = True -> setup___fix_lens_light
        """
        if not self.fix_lens_light:
            return ""
        elif self.fix_lens_light:
            return f"__{conf.instance.setup_tag.get('pipeline', 'fix_lens_light')}"

    def shear_from_previous_pipeline(self, index=0):
        """Return the shear _PriorModel_ from a previous pipeline, where:

        1) If the shear was included in the *Source* pipeline and *no_shear* is *False* in the *Mass* object, it is
           returned using this pipeline result as a model.
        2) If the shear was not included in the *Source* pipeline and *no_shear* is *False* in the *Mass* object, it is
            returned as a new *ExternalShear* PriorModel.
        3) If *no_shear* is *True* in the *Mass* object, it is returned as None and omitted from the lens model.
        """
        if not self.setup_mass.no_shear:
            if af.last[index].model.galaxies.lens.shear is not None:
                return af.last[index].model.galaxies.lens.shear
            else:
                return ag.mp.ExternalShear
        else:
            return None


class SLaM:
    def __init__(
        self,
        folders: [str] = None,
        redshift_lens: float = 0.5,
        redshift_source: float = 1.0,
        setup_hyper: ag_setup.SetupHyper = None,
        pipeline_source_parametric: SLaMPipelineSourceParametric = None,
        pipeline_source_inversion: SLaMPipelineSourceInversion = None,
        pipeline_light: SLaMPipelineLight = None,
        pipeline_mass: SLaMPipelineMass = None,
    ):
        """

        Parameters
        ----------
        folders : [str] or None
            A list of folders that the output of the pipeline are output into before the pipeline name, tags and
            phase folders.
        redshift_lens : float
            The redshift of the lens galaxy used by the pipeline for converting arc-seconds to kpc, masses to solMass,
            etc.
        redshift_source : float
            The redshift of the source galaxy used by the pipeline for converting arc-seconds to kpc, masses to solMass,
            etc.
        setup_hyper : SetupHyper
            The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
        """

        self.folders = folders
        self.redshift_lens = redshift_lens
        self.redshift_source = redshift_source

        self.setup_hyper = setup_hyper

        self.pipeline_source_parametric = pipeline_source_parametric
        self.pipeline_source_inversion = pipeline_source_inversion

        if self.pipeline_source_inversion is not None:
            self.pipeline_source_inversion.setup_light = (
                self.pipeline_source_parametric.setup_light
            )
            self.pipeline_source_inversion.setup_mass = (
                self.pipeline_source_parametric.setup_mass
            )

        self.pipeline_light = pipeline_light

        if self.pipeline_light is not None:
            if self.pipeline_source_inversion is None:
                self.pipeline_light.setup_source = (
                    self.pipeline_source_parametric.setup_source
                )
            else:
                self.pipeline_light.setup_source = (
                    self.pipeline_source_inversion.setup_source
                )

        self.pipeline_mass = pipeline_mass

        if self.pipeline_light is not None:
            self.pipeline_mass.setup_source = self.pipeline_light.setup_source
        else:
            if self.pipeline_source_inversion is None:
                self.pipeline_mass.setup_source = (
                    self.pipeline_source_parametric.setup_source
                )
            else:
                self.pipeline_mass.setup_source = (
                    self.pipeline_source_inversion.setup_source
                )

    @property
    def source_parametric_tag(self):
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to.
        """

        setup_tag = conf.instance.setup_tag.get("source", "source")
        hyper_tag = f"__{self.setup_hyper.tag}" if self.setup_hyper is not None else ""

        if hyper_tag == "__":
            hyper_tag = ""

        source_tag = (
            f"__{self.pipeline_source_parametric.setup_source.tag}"
            if self.pipeline_source_parametric.setup_source is not None
            else ""
        )
        if self.pipeline_light is not None:
            light_tag = (
                f"__{self.pipeline_source_parametric.setup_light.tag}"
                if self.pipeline_source_parametric.setup_light is not None
                else ""
            )
        else:
            light_tag = ""
        mass_tag = (
            f"__{self.pipeline_source_parametric.setup_mass.tag}"
            if self.pipeline_source_parametric.setup_mass is not None
            else ""
        )
        return f"{setup_tag}{hyper_tag}{light_tag}{mass_tag}{source_tag}"

    @property
    def source_inversion_tag(self):
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to.
        """

        setup_tag = conf.instance.setup_tag.get("source", "source")
        hyper_tag = f"__{self.setup_hyper.tag}" if self.setup_hyper is not None else ""

        if hyper_tag == "__":
            hyper_tag = ""

        source_tag = (
            f"__{self.pipeline_source_inversion.setup_source.tag}"
            if self.pipeline_source_inversion.setup_source is not None
            else ""
        )
        if self.pipeline_light is not None:
            light_tag = (
                f"__{self.pipeline_source_inversion.setup_light.tag}"
                if self.pipeline_source_inversion.setup_light is not None
                else ""
            )
        else:
            light_tag = ""
        mass_tag = (
            f"__{self.pipeline_source_inversion.setup_mass.tag}"
            if self.pipeline_source_inversion.setup_mass is not None
            else ""
        )

        return f"{setup_tag}{hyper_tag}{light_tag}{mass_tag}{source_tag}"

    @property
    def source_tag(self):
        if self.pipeline_source_inversion is None:
            return self.source_parametric_tag
        return self.source_inversion_tag

    @property
    def mass_tag(self):
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to.
        """

        setup_tag = conf.instance.setup_tag.get("mass", "mass")
        hyper_tag = f"__{self.setup_hyper.tag}" if self.setup_hyper is not None else ""

        if hyper_tag == "__":
            hyper_tag = ""

        source_tag = (
            f"__{self.pipeline_mass.setup_source.tag}"
            if self.pipeline_mass.setup_source is not None
            else ""
        )
        if self.pipeline_light is not None:
            light_tag = (
                f"__{self.pipeline_mass.setup_light.tag}"
                if self.pipeline_mass.setup_light is not None
                else ""
            )
        else:
            light_tag = ""
        mass_tag = (
            f"__{self.pipeline_mass.setup_mass.tag}"
            if self.pipeline_mass.setup_mass is not None
            else ""
        )

        return f"{setup_tag}{hyper_tag}{light_tag}{mass_tag}{source_tag}"

    def lens_from_light_pipeline_for_mass_pipeline(self, redshift_lens, mass, shear):
        """Setup the lens model for a Mass pipeline using the previous pipeline and phase results.

        The lens light model is not specified by the Mass pipeline, so the Light pipelines are used to
        determine this. This function returns a _GalaxyModel_ for the lens, where:

        1) The lens light model uses the light model of the Light pipeline.
        2) The lens light is returned as a model if *fix_lens_light* is *False, an instance if *True*.

        Parameters
        ----------
        redshift_lens : float
            The redshift of the lens galaxy.
        mass : ag.MassProfile
            The mass model of the len galaxy.
        shear : ag.ExternalShear
            The _ExternalShear_ of the lens galaxy.
        """

        if self.pipeline_mass.fix_lens_light:

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

    def source_from_source_pipeline_for_light_pipeline(self):
        """Setup the source model for a Light pipeline using the previous pipeline and phase results.

        The source light model is not specified by the Light pipeline, so the Source pipelines are used to
        determine whether the source model is parametric or an inversion. This behaviour can be customized in SLaM
        pipelines by replacing this method with the *source_from_previous_pipeline_model_or_instance* method of the
        SLaM class.

        The source is returned as an instance for the Light pipeline, irrespective of whether it is parametric or an
        Inversion. This is because this phase fits the lens light, and thus does not depend strongly on the source.
        """
        return self.source_from_previous_pipeline_model_or_instance(
            source_as_model=False, index=0
        )

    def source_from_source_pipeline_for_mass_pipeline(self, index=0):
        """Setup the source model for a Mass pipeline using the previous pipeline and phase results.

        The source light model is not specified by the pipeline Mass pipeline (e.g. the previous pipelines are used to
        determine whether the source model is parametric or an inversion.

        The source is returned as a model if it is parametric (given it must be updated to properly compute a new mass
        model) whereas inversions are returned as an instance (as they have sufficient flexibility to typically not
        required updating). This behaviour can be customized in SLaM pipelines by replacing this method with the
        *source_from_previous_pipeline_model_or_instance* method of the SLaM class.
        """
        if self.pipeline_source_inversion is None:
            return self.source_from_previous_pipeline_model_or_instance(
                source_as_model=True, index=index
            )
        else:
            return self.source_from_previous_pipeline_model_or_instance(
                source_as_model=False, index=index
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

        if self.setup_hyper.hyper_galaxies:

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

        if self.pipeline_source_inversion is None:

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
