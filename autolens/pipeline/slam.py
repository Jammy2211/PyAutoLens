import autofit as af
import autogalaxy as ag
from autoconf import conf
from autoarray.inversion import pixelizations as pix, regularization as reg
from autogalaxy.profiles import light_profiles as lp, mass_profiles as mp
from autogalaxy.pipeline import setup as ag_setup
from autolens.pipeline import setup

from typing import Union


class AbstractSLaMPipeline:
    def __init__(
        self,
        setup_light: Union[
            ag_setup.SetupLightParametric, ag_setup.SetupLightInversion
        ] = None,
        setup_mass: Union[setup.SetupMassTotal, setup.SetupMassLightDark] = None,
        setup_source: Union[
            setup.SetupSourceParametric, setup.SetupSourceInversion
        ] = None,
    ):
        """
        Abstract class for storing a `SLaMPipeline` object, which contains the `Setup` objects for a given Source,
        Light and Mass (SLaM) pipeline.

        The SLaM pipelines are template pipelines used by PyAutoLens (see `autolens_workspace/slam`) which break the
        model-fitting of a strong lens down into the following 3+ linked pipelines:

        1) Source: Obtain an accurate source model (using parametric `LightProfile`'s and / or an `Inversion`.
        2) Light: Obtain an accurate lens light model (using parametric `LightProfile`'s).
        3) Mass: Obtain an accurate mass model (using a `MassProfile` representing the total mass distribution or
           decomposed `MassProfile`'s representing the light and dark matter).

        Parameters
        ----------
        setup_light : SetupLightParametric
            The setup of the light profile modeling (e.g. for bulge-disk models if they are geometrically aligned).
        setup_mass : SetupMassTotal or SetupMassLightDark
            The setup of the mass modeling (e.g. if a constant mass to light ratio is used).
        setup_source : SetupSourceParametric or SetupSourceInversion
            The setup of the source analysis (e.g. the `LightProfile`, `Pixelization` or `Regularization` used).
        """

        self.setup_light = setup_light
        self.setup_mass = setup_mass
        self.setup_source = setup_source


class SLaMPipelineSourceParametric(AbstractSLaMPipeline):
    def __init__(
        self,
        setup_light: ag_setup.SetupLightParametric = None,
        setup_mass: setup.SetupMassTotal = None,
        setup_source: setup.SetupSourceParametric = None,
    ):
        """
        Abstract class for parametric source `SLaMPipeline` object, which contains the `Setup` objects for a given
        Source, Light and Mass (SLaM) pipeline.

        This object contains the setups for fits in a parametric source pipeline, using `LightProile` `PriorModel`'s to
        fit the source. The lens galaxy light and mass model-fits can be customized, with defaults using an
        `EllipticalSersic` bulge, `EllipticalExponential` disk and `EllipticalIsothermal` mass.

        The SLaM pipelines are template pipelines used by PyAutoLens (see `autolens_workspace/slam`) which break the
        model-fitting of a strong lens down into the following 3+ linked pipelines:

        1) Source: Obtain an accurate source model (using parametric `LightProfile`'s and / or an `Inversion`.
        2) Light: Obtain an accurate lens light model (using parametric `LightProfile`'s).
        3) Mass: Obtain an accurate mass model (using a `MassProfile` representing the total mass distribution or
           decomposed `MassProfile`'s representing the light and dark matter).

        Parameters
        ----------
        setup_light : SetupLightParametric
            The setup of the light profile modeling (e.g. for bulge-disk models if they are geometrically aligned).
        setup_mass : SetupMassTotal
            The setup of the mass modeling (e.g. if a constant mass to light ratio is used).
        setup_source : SetupSourceParametric
            The setup of the source analysis (e.g. the `LightProfile`, `Pixelization` or `Regularization` used).
        """
        if setup_light is None:
            setup_light = ag_setup.SetupLightParametric()

        if setup_mass is None:
            setup_mass = setup.SetupMassTotal(mass_prior_model=mp.EllipticalIsothermal)

        if setup_source is None:
            setup_source = setup.SetupSourceParametric()

        super().__init__(
            setup_light=setup_light, setup_mass=setup_mass, setup_source=setup_source
        )


class SLaMPipelineSourceInversion(AbstractSLaMPipeline):
    def __init__(self, setup_source: setup.SetupSourceInversion = None):
        """
        Abstract class for an inversion source `SLaMPipeline` object, which contains the `Setup` objects for a given
        Source, Light and Mass (SLaM) pipeline.

        This object contains the setups for fits in a inversion source pipeline, using `Pixelization` and
        `Regularization` `PriorModel`'s to fit the  source. The lens galaxy light and mass model-fits assume the
        models fitted in a previous parametric source pipeline, using the results to set their parameter and priors.

        The SLaM pipelines are template pipelines used by PyAutoLens (see `autolens_workspace/slam`) which break the
        model-fitting of a strong lens down into the following 3+ linked pipelines:

        1) Source: Obtain an accurate source model (using parametric `LightProfile`'s and / or an `Inversion`.
        2) Light: Obtain an accurate lens light model (using parametric `LightProfile`'s).
        3) Mass: Obtain an accurate mass model (using a `MassProfile` representing the total mass distribution or
           decomposed `MassProfile`'s representing the light and dark matter).

        Parameters
        ----------
        setup_source : SetupSourceInversion
            The setup of the source analysis (e.g. the `LightProfile`, `Pixelization` or `Regularization` used).
        """
        if setup_source is None:
            setup_source = setup.SetupSourceInversion(
                pixelization_prior_model=pix.Rectangular,
                regularization_prior_model=reg.Constant,
            )

        super().__init__(setup_source=setup_source)


class SLaMPipelineLightParametric(AbstractSLaMPipeline):
    def __init__(self, setup_light: ag_setup.SetupLightParametric = None):
        """
        Abstract class for a parametric light `SLaMPipeline` object, which contains the `Setup` objects for a given
        Source, Light and Mass (SLaM) pipeline.

        The pipeline this object contains the setups for fits in a parametric light pipeline, where `LightProile`
        `PriorModel`'s fit the lens's light. The lens galaxy mass and source galaxy light model-fits assume the models
        fitted in previous source pipelines, using the results to set their parameter and priors.

        The SLaM pipelines are template pipelines used by PyAutoLens (see `autolens_workspace/slam`) which break the
        model-fitting of a strong lens down into the following 3+ linked pipelines:

        1) Source: Obtain an accurate source model (using parametric `LightProfile`'s and / or an `Inversion`.
        2) Light: Obtain an accurate lens light model (using parametric `LightProfile`'s).
        3) Mass: Obtain an accurate mass model (using a `MassProfile` representing the total mass distribution or
           decomposed `MassProfile`'s representing the light and dark matter).

        Parameters
        ----------
        setup_light : SetupLightParametric
            The setup of the light profile modeling (e.g. for bulge-disk models if they are geometrically aligned).
        """
        if setup_light is None:
            setup_light = ag_setup.SetupLightParametric()

        super().__init__(setup_light=setup_light)


class SLaMPipelineMass(AbstractSLaMPipeline):
    def __init__(
        self,
        setup_mass: ag_setup.AbstractSetupMass = None,
        setup_smbh: ag_setup.SetupSMBH = None,
        light_is_model=True,
    ):
        """
        Abstract class for a mass `SLaMPipeline` object, which contains the `Setup` objects for a given Source, Light
        and Mass (SLaM) pipeline.

        The pipeline this object contains the setups for fits in a total mass or light_dark mass pipeline, where
        `MassProile` or `LightMassProfile` `PriorModel`'s fit the lens's mmass. The lens galaxy light and source galaxy
        light models assume those fitted in previous source and light pipelines, using the results to set their
        parameter and priors.

        The SLaM pipelines are template pipelines used by PyAutoLens (see `autolens_workspace/slam`) which break the
        model-fitting of a strong lens down into the following 3+ linked pipelines:

        1) Source: Obtain an accurate source model (using parametric `LightProfile`'s and / or an `Inversion`.
        2) Light: Obtain an accurate lens light model (using parametric `LightProfile`'s).
        3) Mass: Obtain an accurate mass model (using a `MassProfile` representing the total mass distribution or
           decomposed `MassProfile`'s representing the light and dark matter).

        Parameters
        ----------
        setup_mass : SetupMassTotal or SetupMassLightDark
            The setup of the mass modeling (e.g. if a constant mass to light ratio is used).
        setup_smbh : SetupSMBH
            The setup of the super-massive black hole modeling (e.g. its `MassProfile` and if its centre is fixed).
        """
        if setup_mass is None:
            setup_mass = setup.SetupMassTotal()

        super().__init__(setup_mass=setup_mass)

        self.setup_smbh = setup_smbh
        self.light_is_model = light_is_model

    @property
    def light_is_model_tag(self) -> str:
        """
        Tag for if the lens light of the mass pipeline and / or phase are fixed to a previous estimate, or varied
        during he analysis, to customize phase names.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        light_is_model = `False` -> setup__
        light_is_model = `True` -> setup___light_is_model
        """
        if self.light_is_model:
            return f"__{conf.instance['notation']['setup_tags']['pipeline']['light_is_model']}"
        return f"__{conf.instance['notation']['setup_tags']['pipeline']['light_is_instance']}"

    @property
    def smbh_prior_model(self):

        if self.setup_smbh is not None:
            return self.setup_smbh.smbh_from_centre(
                centre=af.last.instance.galaxies.lens.sersic.centre
            )

    def shear_from_previous_pipeline(self, index=0):
        """Return the shear `PriorModel` from a previous pipeline, where:

        1) If the shear was included in the *Source* pipeline and `with_shear` is `False` in the `Mass` object, it is
           returned using this pipeline result as a model.
        2) If the shear was not included in the *Source* pipeline and *with_shear* is `False` in the `Mass` object,
            it is returned as a new *ExternalShear* PriorModel.
        3) If `with_shear` is `True` in the `Mass` object, it is returned as None and omitted from the lens model.
        """
        if self.setup_mass.with_shear:
            if af.last[index].model.galaxies.lens.shear is not None:
                return af.last[index].model.galaxies.lens.shear
            else:
                return ag.mp.ExternalShear


class SLaM:
    def __init__(
        self,
        path_prefix: str = None,
        redshift_lens: float = 0.5,
        redshift_source: float = 1.0,
        setup_hyper: setup.SetupHyper = None,
        pipeline_source_parametric: SLaMPipelineSourceParametric = None,
        pipeline_source_inversion: SLaMPipelineSourceInversion = None,
        pipeline_light_parametric: SLaMPipelineLightParametric = None,
        pipeline_mass: SLaMPipelineMass = None,
        setup_subhalo: setup.SetupSubhalo = None,
    ):
        """

        Parameters
        ----------
        path_prefix : str or None
            The prefix of folders between the output path of the pipeline and the pipeline name, tags and phase folders.
        redshift_lens : float
            The redshift of the lens galaxy used by the pipeline for converting arc-seconds to kpc, masses to solMass,
            etc.
        redshift_source : float
            The redshift of the source galaxy used by the pipeline for converting arc-seconds to kpc, masses to solMass,
            etc.
        setup_hyper : SetupHyper
            The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
        pipeline_source_parametric : SLaMPipelineSourceInversion
            Contains the `Setup`'s used in the parametric source pipeline of the SLaM pipelines.
        pipeline_source_inversion : SLaMPipelineSourceInversion
            Contains the `Setup`'s used in the inversion source pipeline of the SLaM pipelines.
        pipeline_light_parametric : SLaMPipelineLightParametric
            Contains the `Setup`'s used in the parametric light pipeline of the SLaM pipelines.
        pipeline_mass : SLaMPipelineLightParametric
            Contains the `Setup`'s used in the mass pipeline of the SLaM pipelines.
        setup_subhalo : SetupSubhalo
            The setup of a subhalo in the mass model, if included.
        """

        self.path_prefix = path_prefix
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

        self.pipeline_light = pipeline_light_parametric

        if self.pipeline_light is not None:

            self.pipeline_light.setup_mass = self.pipeline_source_parametric.setup_mass

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
            self.pipeline_mass.setup_light = self.pipeline_light.setup_light
        else:
            if self.pipeline_source_inversion is None:
                self.pipeline_mass.setup_source = (
                    self.pipeline_source_parametric.setup_source
                )
            else:
                self.pipeline_mass.setup_source = (
                    self.pipeline_source_inversion.setup_source
                )

        self.setup_subhalo = setup_subhalo

    @property
    def source_parametric_tag(self) -> str:
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to."""

        setup_tag = conf.instance["notation"]["setup_tags"]["names"]["source"]
        hyper_tag = (
            f"__{self.setup_hyper.tag_no_fixed}" if self.setup_hyper is not None else ""
        )

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
    def source_inversion_tag(self) -> str:
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to."""

        setup_tag = conf.instance["notation"]["setup_tags"]["names"]["source"]
        hyper_tag = (
            f"__{self.setup_hyper.tag_no_fixed}" if self.setup_hyper is not None else ""
        )

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
    def source_tag(self) -> str:
        if self.pipeline_source_inversion is None:
            return self.source_parametric_tag
        return self.source_inversion_tag

    @property
    def light_parametric_tag(self) -> str:
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to."""

        setup_tag = conf.instance["notation"]["setup_tags"]["names"]["light"]
        hyper_tag = f"__{self.setup_hyper.tag}" if self.setup_hyper is not None else ""

        if hyper_tag == "__":
            hyper_tag = ""

        source_tag = (
            f"__{self.pipeline_light.setup_source.tag}"
            if self.pipeline_light.setup_source is not None
            else ""
        )

        light_tag = (
            f"__{self.pipeline_light.setup_light.tag}"
            if self.pipeline_light.setup_light is not None
            else ""
        )

        mass_tag = (
            f"__{self.pipeline_light.setup_mass.tag}"
            if self.pipeline_light.setup_mass is not None
            else ""
        )

        return f"{setup_tag}{hyper_tag}{light_tag}{mass_tag}{source_tag}"

    @property
    def mass_tag(self) -> str:
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to."""

        setup_tag = conf.instance["notation"]["setup_tags"]["names"]["mass"]
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

    def lens_from_light_parametric_pipeline_for_mass_total_pipeline(
        self, mass: af.PriorModel, shear: af.PriorModel
    ) -> ag.GalaxyModel:
        """Setup the lens model for a `Mass` pipeline using the previous `Light` pipeline and phase results.

        The lens light model is not specified by the `Mass` pipeline, so the `Light` pipelines are used to
        determine this. This function returns a `GalaxyModel` for the lens, where:

        1) The lens light model uses the light model of the Light pipeline.
        2) The lens light is returned as a model if *light_is_model* is *False, an instance if `True`.

        Parameters
        ----------
        redshift_lens : float
            The redshift of the lens galaxy.
        mass : ag.MassProfile
            The mass model of the len galaxy.
        shear : ag.ExternalShear
            The `ExternalShear` of the lens galaxy.

        Returns
        -------
        ag.GalaxyModel
            Contains the `PriorModel`'s of the lens's light, mass, the shear, etc.
        """

        if not self.pipeline_mass.light_is_model:

            return ag.GalaxyModel(
                redshift=self.redshift_lens,
                bulge=af.last.instance.galaxies.lens.bulge,
                disk=af.last.instance.galaxies.lens.disk,
                envelope=af.last.instance.galaxies.lens.envelope,
                mass=mass,
                shear=shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            )

        else:

            return ag.GalaxyModel(
                redshift=self.redshift_lens,
                bulge=af.last.model.galaxies.lens.bulge,
                disk=af.last.model.galaxies.lens.disk,
                envelope=af.last.model.galaxies.lens.envelope,
                mass=mass,
                shear=shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            )

    def lens_for_subhalo_pipeline(self, index: int = 0) -> ag.GalaxyModel:
        """
        Pass the lens `PriorModel` as a `model` or `instance` from the `MassPipeline` to the `SubhaloPipeline`.

        Parameters
        ----------
        index : int
            The index of the previous phase from which the `PriorModel`'s are passed.

        Returns
        -------
        ag.GalaxyModel
            Contains the `PriorModel`'s of the lens's mass, the shear, etc.
        """

        if self.setup_subhalo.mass_is_model:

            return ag.GalaxyModel(
                redshift=self.redshift_lens,
                mass=af.last[index].model.galaxies.lens.mass,
                shear=af.last[index].model.galaxies.lens.shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            )

        else:

            return ag.GalaxyModel(
                redshift=self.redshift_lens,
                mass=af.last[index].instance.galaxies.lens.mass,
                shear=af.last[index].instance.galaxies.lens.shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            )

    def _source_parametric_from_previous_pipeline(
        self, source_is_model: bool = True, index: int = 0
    ) -> ag.GalaxyModel:
        """
         Pass a parametric source `PriorModel` as a `model` or `instance` from a previous pipeline.

        Parameters
        ----------
        source_is_model : bool
            If `True`, the source is passed as a `model` and fitted for by the pipeline. If `False` it is
            passed as an `instance` with fixed parameters.
        index : int
            The index of the previous phase the source results are passed from.

        Returns
        -------
        ag.GalaxyModel
            Contains the `PriorModel`'s of the source's bulge, disk, etc.
        """
        hyper_galaxy = self.setup_hyper.hyper_galaxy_source_from_previous_pipeline(
            index=index
        )

        if source_is_model:

            return ag.GalaxyModel(
                redshift=self.redshift_source,
                bulge=af.last[index].model.galaxies.source.bulge,
                disk=af.last[index].model.galaxies.source.disk,
                envelope=af.last[index].model.galaxies.source.envelope,
                hyper_galaxy=hyper_galaxy,
            )

        else:

            return ag.GalaxyModel(
                redshift=self.redshift_source,
                bulge=af.last[index].instance.galaxies.source.bulge,
                disk=af.last[index].instance.galaxies.source.disk,
                envelope=af.last[index].instance.galaxies.source.envelope,
                hyper_galaxy=hyper_galaxy,
            )

    def _source_inversion_from_previous_pipeline(
        self, source_is_model: bool = False, index: int = 0
    ) -> ag.GalaxyModel:
        """
         Pass an inversion source `PriorModel` as a `model` or `instance` from a previous pipeline.

        Parameters
        ----------
        source_is_model : bool
            If `True`, the source is passed as a `model` and fitted for by the pipeline. If `False` it is
            passed as an `instance` with fixed parameters.
        index : int
            The index of the previous phase the source results are passed from.

        Returns
        -------
        ag.GalaxyModel
            Contains the `PriorModel`'s of the source's pixelization, regularization, etc.
        """
        hyper_galaxy = self.setup_hyper.hyper_galaxy_source_from_previous_pipeline(
            index=index
        )

        if source_is_model:

            return ag.GalaxyModel(
                redshift=self.redshift_source,
                pixelization=af.last[
                    index
                ].hyper_combined.instance.galaxies.source.pixelization,
                regularization=af.last[
                    index
                ].hyper_combined.model.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )

        else:

            return ag.GalaxyModel(
                redshift=self.redshift_source,
                pixelization=af.last[
                    index
                ].hyper_combined.instance.galaxies.source.pixelization,
                regularization=af.last[
                    index
                ].hyper_combined.instance.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )

    def source_from_previous_pipeline(
        self, source_is_model: bool = False, index: int = 0
    ) -> ag.GalaxyModel:
        """
        Setup the source model using the previous pipeline and phase results.

        The source light model is not specified by the pipeline light and mass pipelines (e.g. the previous pipelines
        are used to determine whether the source model is parametric or an inversion).

        The source can be returned as an `instance` or `model`, depending on the optional input. The default SLaM
        pipelines return parametric sources as a model (give they must be updated to properly compute a new mass
        model) and return inversions as an instance (as they have sufficient flexibility to typically not required
        updating). They use the *source_from_pevious_pipeline* method of the SLaM class to do this.

        The pipeline tool af.last is required to locate the previous source model, which requires an index based
        on the pipelines that have run. For example, if the source model you wish to load from is 3 phases back
        (perhaps because there were multiple phases in a Light pipeline preivously) this index should be 2.

        Parameters
        ----------
        source_is_model : bool
            If `True` the source is returned as a *model* where the parameters are fitted for using priors of the
            phase result it is loaded from. If `False`, it is an instance of that phase's result.
        index : integer
            The index (counting backwards from this phase) of the phase result used to setup the source.
        """

        if self.pipeline_source_inversion is None:

            return self._source_parametric_from_previous_pipeline(
                source_is_model=source_is_model, index=index
            )

        else:

            return self._source_inversion_from_previous_pipeline(
                source_is_model=source_is_model, index=index
            )

    def source_from_previous_pipeline_model_if_parametric(self, index=0):
        """Setup the source model for a Mass pipeline using the previous pipeline and phase results.

        The source light model is not specified by the pipeline Mass pipeline (e.g. the previous pipelines are used to
        determine whether the source model is parametric or an inversion.

        The source is returned as a model if it is parametric (given it must be updated to properly compute a new mass
        model) whereas inversions are returned as an instance (as they have sufficient flexibility to typically not
        required updating). This behaviour can be customized in SLaM pipelines by replacing this method with the
        *source_from_previous_pipeline_model_or_instance* method of the SLaM class.
        """
        if self.pipeline_source_inversion is None:
            return self.source_from_previous_pipeline(source_is_model=True, index=index)
        return self.source_from_previous_pipeline(source_is_model=False, index=index)

    def source_for_subhalo_pipeline(self, index=0):
        return self.source_from_previous_pipeline(
            source_is_model=self.setup_subhalo.source_is_model, index=index
        )
