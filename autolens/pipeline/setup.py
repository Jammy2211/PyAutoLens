import autofit as af
from autoconf import conf
from autogalaxy.pipeline import setup
from autoarray.inversion import pixelizations as pix, regularization as reg
from autogalaxy.profiles import (
    light_profiles as lp,
    mass_profiles as mp,
    light_and_mass_profiles as lmp,
)
from autogalaxy.galaxy import galaxy as g

from typing import Union


class SetupHyper(setup.SetupHyper):
    def __init__(
        self,
        hyper_galaxies_lens: bool = False,
        hyper_galaxies_source: bool = False,
        hyper_image_sky: bool = False,
        hyper_background_noise: bool = False,
        hyper_galaxy_phase_first: bool = False,
        hyper_fixed_after_source: bool = False,
        hyper_galaxies_search: af.NonLinearSearch = None,
        inversion_search: af.NonLinearSearch = None,
        hyper_combined_search: af.NonLinearSearch = None,
        evidence_tolerance: float = None,
    ):
        """
        The hyper setup of a pipeline, which controls how hyper-features in PyAutoLens template pipelines run,
        for example controlling whether hyper galaxies are used to scale the noise and the non-linear searches used
        in these phases.

        Users can write their own pipelines which do not use or require the *SetupHyper* class.

        This class enables pipeline tagging, whereby the hyper setup of the pipeline is used in the template pipeline
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
        hyper_galaxy_phase_first : bool
            If True, the hyper-galaxy phase which scales the noise map is performed before the inversion phase, else
            it is performed after.
        hyper_fixed_after_source : bool
            If `True`, the hyper parameters are fixed and not updated after a desnated pipeline in the analysis. For
            the `SLaM` pipelines this is after the `SourcePipeline`. This allow Bayesian model comparison to be
            performed objected between later phases in a pipeline.
        hyper_galaxies_search : af.NonLinearSearch or None
            The `NonLinearSearch` used by every hyper-galaxies phase.
        inversion_search : af.NonLinearSearch or None
            The `NonLinearSearch` used by every inversion phase.
        hyper_combined_search : af.NonLinearSearch or None
            The `NonLinearSearch` used by every hyper combined phase.
        evidence_tolerance : float
            The evidence tolerance of the non-linear searches used in the hyper phases, whereby higher values will
            lead them to end earlier at the expense of accuracy.
        """
        if hyper_galaxies_lens or hyper_galaxies_source:
            hyper_galaxies = True
        else:
            hyper_galaxies = False

        super().__init__(
            hyper_galaxies=hyper_galaxies,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            hyper_galaxy_phase_first=hyper_galaxy_phase_first,
            hyper_galaxies_search=hyper_galaxies_search,
            inversion_search=inversion_search,
            hyper_combined_search=hyper_combined_search,
            evidence_tolerance=evidence_tolerance,
        )

        self.hyper_galaxies_lens = hyper_galaxies_lens
        self.hyper_galaxies_source = hyper_galaxies_source

        if self.hyper_galaxies_lens or self.hyper_galaxies_source:
            self.hyper_galaxy_names = []

        if self.hyper_galaxies_lens:
            self.hyper_galaxy_names.append("lens")

        if self.hyper_galaxies_source:
            self.hyper_galaxy_names.append("source")

        self.hyper_fixed_after_source = hyper_fixed_after_source

    @property
    def tag(self):
        """
        Tag the pipeline according to the setup of the hyper feature, which customizes the pipeline output paths.

        This includes tags for whether hyper-galaxies are used to scale the noise-map and whether the background sky or
        noise are fitted for by the pipeline.

        For the default configuration files in `config/notation/setup_tags.ini` example tags appear as:

        - hyper[galaxies_lens__bg_sky]
        - hyper[bg_sky__bg_noise__fixed_after_source]
        """
        if not any(
            [self.hyper_galaxies, self.hyper_image_sky, self.hyper_background_noise]
        ):
            return ""

        return (
            f"{self.component_name}["
            f"{self.hyper_galaxies_tag}"
            f"{self.hyper_image_sky_tag}"
            f"{self.hyper_background_noise_tag}"
            f"{self.hyper_fixed_after_source_tag}]"
        )

    @property
    def tag_no_fixed(self):
        """
        Tag the pipeline according to the setup of the hyper feature, which customizes the pipeline output paths.

        This tag is the same as the `tag` property but does not include the `hyper_fixed_after_source_tag`, and is
        used to tag pipelines after the `source` pipeline that this tag specifically tags.

        For the default configuration files in `config/notation/setup_tags.ini` example tags appear as:

        - hyper[galaxies__bg_sky]
        - hyper[bg_sky__bg_noise]
        """
        if not any(
            [self.hyper_galaxies, self.hyper_image_sky, self.hyper_background_noise]
        ):
            return ""

        return (
            f"{self.component_name}["
            f"{self.hyper_galaxies_tag}"
            f"{self.hyper_image_sky_tag}"
            f"{self.hyper_background_noise_tag}]"
        )

    @property
    def hyper_galaxies_tag(self) -> str:
        """
        Tag for if hyper-galaxies are used in a hyper pipeline to scale the noise-map duing model fitting, which
        customizes the pipeline's output paths.

        The tag is generated separately for the lens and souce galaxies and depends on the `hyper_galaxies_lens` and
        `hyper_galaxies_source` bools of the `SetupHyper`.

        For the the default configs tagging is performed as follows:

        - `hyper_galaxies_lens=False`, `hyper_galaxies_source=False` -> No Tag
        - `hyper_galaxies_lens=True`, `hyper_galaxies_source=False` -> hyper[galaxies_lens]
        - `hyper_galaxies_lens=False`, `hyper_galaxies_source=True` -> hyper[galaxies_source]
        - `hyper_galaxies_lens=True`, `hyper_galaxies_source=True` -> hyper[galaxies_lens_source]

        This is used to generate an overall tag in `tag`.
        """
        if not self.hyper_galaxies:
            return ""

        hyper_galaxies_tag = conf.instance["notation"]["setup_tags"]["hyper"][
            "hyper_galaxies"
        ]

        if self.hyper_galaxies_lens:
            hyper_galaxies_lens_tag = f"_{conf.instance['notation']['setup_tags']['hyper']['hyper_galaxies_lens']}"
        else:
            hyper_galaxies_lens_tag = ""

        if self.hyper_galaxies_source:
            hyper_galaxies_source_tag = f"_{conf.instance['notation']['setup_tags']['hyper']['hyper_galaxies_source']}"
        else:
            hyper_galaxies_source_tag = ""

        return (
            f"{hyper_galaxies_tag}{hyper_galaxies_lens_tag}{hyper_galaxies_source_tag}"
        )

    @property
    def hyper_fixed_after_source_tag(self) -> str:
        """
        Tag for if the hyper parameters are held fixed after the source pipeline.

        For the the default configs tagging is performed as follows:

        hyper_fixed_after_source = `False` -> No Tag
        hyper_fixed_after_source = `True` -> hyper[other_tags_fixed]
        """
        if not self.hyper_fixed_after_source:
            return ""
        elif self.hyper_fixed_after_source:
            return f"__{conf.instance['notation']['setup_tags']['hyper']['hyper_fixed_after_source']}"

    def hyper_galaxy_lens_from_previous_pipeline(
        self, index=0, noise_factor_is_model=False
    ):
        """
        Returns the `HyperGalaxy` `PriorModel` from a previous pipeline or phase of the lens galaxy in a template
        PyAutoLens pipeline.

        The `HyperGalaxy` is extracted from the `hyper_combined` phase of the previous pipeline, and by default has its
        parameters passed as instance's which are fixed in the next phase.

        If `noise_factor_is_model` is `True` the `noise_factor` parameter of the `HyperGalaxy` is passed as a model and
        fitted for by the phase. This is typically used when the lens model complexity is updated and it is possible
        that the noise-scaling performed in the previous phase (using a simpler lens light model) over-scales the
        noise for the new more complex light profile.

        Parameters
        ----------
        index : int
            The index of the previous phase the `HyperGalaxy` `PriorModel` is passed from.
        noise_factor_is_model : bool
            If `True` the `noise_factor` of the `HyperGalaxy` is passed as a `model`, else it is passed as an
            `instance`.

        Returns
        -------
        af.PriorModel(g.HyperGalaxy)
            The hyper-galaxy that is passed to the next phase.
        """
        if self.hyper_galaxies:

            hyper_galaxy = af.PriorModel(g.HyperGalaxy)

            if noise_factor_is_model:

                hyper_galaxy.noise_factor = af.last[
                    index
                ].hyper_combined.model.galaxies.lens.hyper_galaxy.noise_factor

            else:

                hyper_galaxy.noise_factor = af.last[
                    index
                ].hyper_combined.instance.galaxies.lens.hyper_galaxy.noise_factor

            hyper_galaxy.contribution_factor = af.last[
                index
            ].hyper_combined.instance.optional.galaxies.lens.hyper_galaxy.contribution_factor
            hyper_galaxy.noise_power = af.last[
                index
            ].hyper_combined.instance.optional.galaxies.lens.hyper_galaxy.noise_power

            return hyper_galaxy

    def hyper_galaxy_source_from_previous_pipeline(
        self, index=0, noise_factor_is_model=False
    ):
        """
        Returns the `HyperGalaxy` `PriorModel` from a previous pipeline or phase of the source galaxy in a template
        PyAutosource pipeline.

        The `HyperGalaxy` is extracted from the `hyper_combined` phase of the previous pipeline, and by default has its
        parameters passed as instance's which are fixed in the next phase.

        If `noise_factor_is_model` is `True` the `noise_factor` parameter of the `HyperGalaxy` is passed as a model and
        fitted for by the phase. This is typically used when the source model complexity is updated and it is possible
        that the noise-scaling performed in the previous phase (using a simpler source light model) over-scales the
        noise for the new more complex light profile.

        Parameters
        ----------
        index : int
            The index of the previous phase the `HyperGalaxy` `PriorModel` is passed from.
        noise_factor_is_model : bool
            If `True` the `noise_factor` of the `HyperGalaxy` is passed as a `model`, else it is passed as an
            `instance`.

        Returns
        -------
        af.PriorModel(g.HyperGalaxy)
            The hyper-galaxy that is passed to the next phase.
        """
        if self.hyper_galaxies:

            hyper_galaxy = af.PriorModel(g.HyperGalaxy)

            if noise_factor_is_model:

                hyper_galaxy.noise_factor = af.last[
                    index
                ].hyper_combined.model.galaxies.source.hyper_galaxy.noise_factor

            else:

                hyper_galaxy.noise_factor = af.last[
                    index
                ].hyper_combined.instance.galaxies.source.hyper_galaxy.noise_factor

            hyper_galaxy.contribution_factor = af.last[
                index
            ].hyper_combined.instance.optional.galaxies.source.hyper_galaxy.contribution_factor
            hyper_galaxy.noise_power = af.last[
                index
            ].hyper_combined.instance.optional.galaxies.source.hyper_galaxy.noise_power

            return hyper_galaxy


class AbstractSetupMass:

    with_shear = None

    @property
    def with_shear_tag(self) -> str:
        """Generate a tag if an `ExternalShear` is included in the mass model of the pipeline  are
        fixedto a previous estimate, or varied during the analysis, to customize pipeline output paths..

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        with_shear = `False` -> setup__with_shear
        with_shear = `True` -> setup___with_shear
        """
        if self.with_shear:
            return f"__{conf.instance['notation']['setup_tags']['mass']['with_shear']}"
        return f"__{conf.instance['notation']['setup_tags']['mass']['no_shear']}"

    @property
    def shear_prior_model(self) -> af.PriorModel:
        """For a SLaM source pipeline, determine the shear model from the with_shear setting."""
        if self.with_shear:
            return af.PriorModel(mp.ExternalShear)


class SetupMassTotal(setup.SetupMassTotal, AbstractSetupMass):
    def __init__(
        self,
        mass_prior_model: af.PriorModel(mp.MassProfile) = mp.EllipticalPowerLaw,
        with_shear=True,
        mass_centre: (float, float) = None,
        align_bulge_mass_centre: bool = False,
    ):
        """
        The setup of the mass modeling in a pipeline for `MassProfile`'s representing the total (e.g. stars + dark
        matter) mass distribution, which controls how PyAutoGalaxy template pipelines run, for example controlling
        assumptions about the bulge-disk model.

        Users can write their own pipelines which do not use or require the `SetupMassTotal` class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        mass_prior_model : af.PriorModel(mp.MassProfile)
            The `MassProfile` fitted by the `Pipeline` (the pipeline must specifically use this option to use this
            mass profile)
        with_shear : bool
            If `True` the `ExternalShear` `PriorModel` is omitted from the galaxy model.
        mass_centre : (float, float) or None
           If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
           non-linear search.
        align_bulge_mass_centre : bool
            If `True` and the galaxy model has both a light and mass component, the function
            `align_centre_of_mass_to_light` can be used to align their centres.
        """

        super().__init__(
            mass_prior_model=mass_prior_model,
            mass_centre=mass_centre,
            align_bulge_mass_centre=align_bulge_mass_centre,
        )

        self.with_shear = with_shear

    @property
    def tag(self) -> str:
        """
        Tag the pipeline according to the setup of the total mass pipeline which customizes the pipeline output paths.

        This includes tags for the `MassProfile` `PriorModel`'s and the alignment of different components in the model.

        For the default configuration files in `config/notation/setup_tags.ini` example tags appear as:

        - mass[total__sie]
        - mass[total__power_law__centre_(0.0,0.0)]
        """
        return (
            f"{self.component_name}[total"
            f"{self.mass_prior_model_tag}"
            f"{self.with_shear_tag}"
            f"{self.mass_centre_tag}"
            f"{self.align_bulge_mass_centre_tag}]"
        )


class SetupMassLightDark(setup.SetupMassLightDark, AbstractSetupMass):
    def __init__(
        self,
        with_shear=True,
        bulge_prior_model: af.PriorModel(lmp.LightMassProfile) = lmp.EllipticalSersic,
        disk_prior_model: af.PriorModel(
            lmp.LightMassProfile
        ) = lmp.EllipticalExponential,
        envelope_prior_model: af.PriorModel(lmp.LightMassProfile) = None,
        mass_centre: (float, float) = None,
        constant_mass_to_light_ratio: bool = False,
        align_bulge_dark_centre: bool = False,
    ):
        """
        The setup of the mass modeling in a pipeline for `MassProfile`'s representing the decomposed light and dark
        mass distributions, which controls how PyAutoGalaxy template pipelines run, for example controlling assumptions
        about the bulge-disk model.

        Users can write their own pipelines which do not use or require the `SetupMassLightDark` class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        with_shear : bool
           If `True` the `ExternalShear` `PriorModel` is omitted from the galaxy model.
        bulge_prior_model : af.PriorModel or al.lmp.LightMassProfile
            The `LightProfile` `PriorModel` used to represent the light distribution of a bulge.
        disk_prior_model : af.PriorModel(al.lmp.LightMassProfile)
            The `LightProfile` `PriorModel` used to represent the light distribution of a disk.
        envelope_prior_model : af.PriorModel(al.lmp.LightMassProfile)
            The `LightProfile` `PriorModel` used to represent the light distribution of a envelope.
        mass_centre : (float, float)
           If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
           non-linear search.
        constant_mass_to_light_ratio : bool
            If True, and the mass model consists of multiple `LightProfile` and `MassProfile` coomponents, the
            mass-to-light ratio's of all components are fixed to one shared value.
        align_bulge_mass_centre : bool
            If True, and the mass model is a decomposed bulge, disk and dark matter model (e.g. EllipticalSersic +
            EllipticalExponential + SphericalNFW), the centre of the bulge and dark matter profiles are aligned.
        """
        super().__init__(
            bulge_prior_model=bulge_prior_model,
            disk_prior_model=disk_prior_model,
            envelope_prior_model=envelope_prior_model,
            mass_centre=mass_centre,
            constant_mass_to_light_ratio=constant_mass_to_light_ratio,
            align_bulge_dark_centre=align_bulge_dark_centre,
        )

        self.with_shear = with_shear

    @property
    def tag(self):
        """
        Tag the pipeline according to the setup of the decomposed light and dark mass pipeline which customizes
        the pipeline output paths.

        This includes tags for the `MassProfile` `PriorModel`'s and the alignment of different components in the model.

        For the default configuration files in `config/notation/setup_tags.ini` example tags appear as:

        - mass[light_dark__bulge_]
        - mass[total_power_law__centre_(0.0,0.0)]
        """
        return (
            f"{self.component_name}[light_dark"
            f"{self.bulge_prior_model_tag}"
            f"{self.disk_prior_model_tag}"
            f"{self.envelope_prior_model_tag}"
            f"{self.constant_mass_to_light_ratio_tag}"
            f"{self.dark_prior_model_tag}"
            f"{self.with_shear_tag}"
            f"{self.mass_centre_tag}"
            f"{self.align_bulge_dark_centre_tag}]"
        )


class SetupSourceParametric(setup.SetupLightParametric):
    def __init__(
        self,
        bulge_prior_model: af.PriorModel(lp.LightProfile) = lp.EllipticalSersic,
        disk_prior_model: af.PriorModel(lp.LightProfile) = None,
        envelope_prior_model: af.PriorModel(lp.LightProfile) = None,
        light_centre: (float, float) = None,
        align_bulge_disk_centre: bool = True,
        align_bulge_disk_elliptical_comps: bool = False,
        align_bulge_envelope_centre: bool = False,
    ):
        """
        The setup of the light modeling in a pipeline, which controls how PyAutoGalaxy template pipelines runs, for
        example controlling assumptions about the bulge-disk model.

        Users can write their own pipelines which do not use or require the *SetupLightParametric* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        bulge_prior_model : af.PriorModel(lp.LightProfile)
            The `LightProfile` `PriorModel` used to represent the light distribution of a bulge.
        disk_prior_model : af.PriorModel(lp.LightProfile)
            The `LightProfile` `PriorModel` used to represent the light distribution of a disk.
        envelope_prior_model : af.PriorModel(lp.LightProfile)
            The `LightProfile` `PriorModel` used to represent the light distribution of a envelope.
        light_centre : (float, float) or None
           If input, a fixed (y,x) centre of the galaxy is used for the light profile model which is not treated as a
            free parameter by the non-linear search.
        align_bulge_disk_centre : bool or None
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            `True` will align the centre of the bulge and disk components and not fit them separately.
        align_bulge_disk_elliptical_comps : bool or None
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            `True` will align the elliptical components the bulge and disk components and not fit them separately.
        align_bulge_envelope_centre : bool or None
            If a bulge + envelope light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the
            galaxy, `True` will align the centre of the bulge and envelope components and not fit them separately.
        """

        super().__init__(
            bulge_prior_model=bulge_prior_model,
            disk_prior_model=disk_prior_model,
            envelope_prior_model=envelope_prior_model,
            light_centre=light_centre,
            align_bulge_disk_centre=align_bulge_disk_centre,
            align_bulge_disk_elliptical_comps=align_bulge_disk_elliptical_comps,
            align_bulge_envelope_centre=align_bulge_envelope_centre,
        )

    @property
    def component_name(self) -> str:
        """
        The name of the source component of a `source` pipeline which preceeds the `Setup` tag contained within square
        brackets.

        For the default configuration files this tag appears as `source[tag]`.

        Returns
        -------
        str
            The component name of the source pipeline.
        """
        return conf.instance["notation"]["setup_tags"]["names"]["source"]


class SetupSourceInversion(setup.SetupLightInversion):
    def __init__(
        self,
        pixelization_prior_model: af.PriorModel(pix.Pixelization),
        regularization_prior_model: af.PriorModel(reg.Regularization),
        inversion_pixels_fixed: float = None,
    ):
        """
        The setup of the inversion source modeling of a pipeline, which controls how PyAutoGalaxy template pipelines run,
        for example controlling the `Pixelization` and `Regularization` used by the `Inversion`.

        Users can write their own pipelines which do not use or require the `SetupLightInversion` class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        pixelization_prior_model : af.PriorModel(pix.Pixelization)
           If the pipeline uses an `Inversion` to reconstruct the galaxy's source, this determines the `Pixelization`
           used.
        regularization_prior_model : af.PriorModel(reg.Regularization)
            If the pipeline uses an `Inversion` to reconstruct the galaxy's source, this determines the `Regularization`
            scheme used.
        inversion_pixels_fixed : float
            The fixed number of source pixels used by a `Pixelization` class that takes as input a fixed number of
            pixels.
        """

        super().__init__(
            pixelization_prior_model=pixelization_prior_model,
            regularization_prior_model=regularization_prior_model,
            inversion_pixels_fixed=inversion_pixels_fixed,
        )

    @property
    def component_name(self) -> str:
        """
        The name of the source component of a `source` pipeline which preceeds the `Setup` tag contained within square
        brackets.

        For the default configuration files this tag appears as `source[tag]`.

        Returns
        -------
        str
            The component name of the source pipeline.
        """
        return conf.instance["notation"]["setup_tags"]["names"]["source"]


class SetupSubhalo(setup.AbstractSetup):
    def __init__(
        self,
        subhalo_prior_model: af.PriorModel(mp.MassProfile) = mp.SphericalNFWMCRLudlow,
        subhalo_search: af.NonLinearSearch = None,
        source_is_model: bool = True,
        mass_is_model: bool = True,
        grid_size: int = 5,
        grid_dimension_arcsec: float = 3.0,
        parallel: bool = False,
        subhalo_instance=None,
    ):
        """
        The setup of a subhalo pipeline, which controls how PyAutoLens template pipelines runs.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        subhalo_search : af.NonLinearSearch
            The search used to sample parameter space in the subhalo pipeline.
        source_is_model : bool
            If `True`, the source is included as a model in the fit (for both `LightProfile` or `Inversion` sources).
            If `False` its parameters are fixed to those inferred in a previous pipeline.
        mass_is_model : bool
            If `True`, the mass is included as a model in the fit. If `False` its parameters are fixed to those
            inferred in a previous pipeline.
        grid_size : int
            The 2D dimensions of the grid (e.g. grid_size x grid_size) that the subhalo search is performed for.
        grid_dimension_arcsec : float
            the arc-second dimensions of the grid in the y and x directions. An input value of 3.0" means the grid in
            all four directions extends to 3.0" giving it dimensions 6.0" x 6.0".
        parallel : bool
            If `True` the `Python` `multiprocessing` module is used to parallelize the fitting over the cpus available
            on the system.
        subhalo_instance : ag.MassProfile
            An instance of the mass-profile used as a fixed model for a subhalo pipeline.
        """

        if subhalo_search is None:
            subhalo_search = af.DynestyStatic(n_live_points=50, walks=5, facc=0.2)

        self.subhalo_prior_model = self._cls_to_prior_model(cls=subhalo_prior_model)

        self.subhalo_search = subhalo_search
        self.source_is_model = source_is_model
        self.mass_is_model = mass_is_model
        self.grid_size = grid_size
        self.grid_dimensions_arcsec = grid_dimension_arcsec
        self.parallel = parallel
        self.subhalo_instance = subhalo_instance

    @property
    def component_name(self) -> str:
        """
        The name of the subhalo component of a `subhalo` pipeline which preceeds the `Setup` tag contained within square
        brackets.

        For the default configuration files this tag appears as `subhalo[tag]`.

        Returns
        -------
        str
            The component name of the subhalo pipeline.
        """
        return conf.instance["notation"]["setup_tags"]["names"]["subhalo"]

    @property
    def tag(self):
        return (
            f"{self.component_name}["
            f"{self.subhalo_prior_model_tag}"
            f"{self.mass_is_model_tag}"
            f"{self.source_is_model_tag}"
            f"{self.grid_size_tag}"
            f"{self.subhalo_centre_tag}"
            f"{self.subhalo_mass_at_200_tag}]"
        )

    @property
    def subhalo_prior_model_tag(self) -> str:
        """
        The tag of the subhalo `PriorModel` the `MassProfile` class given to the `subhalo_prior_model`.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `EllipticalIsothermal` -> sie
        - `EllipticalPowerLaw` -> power_law

        Returns
        -------
        str
            The tag of the subhalo prior model.
        """

        if self.subhalo_prior_model is None:
            return ""

        return f"{conf.instance['notation']['prior_model_tags']['mass'][self.subhalo_prior_model.name]}"

    @property
    def mass_is_model_tag(self) -> str:
        """
        Tags if the lens mass model during the subhalo pipeline is model or instance.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        mass_is_model = `True` -> setup[mass_is_model]
        mass_is_model = `False` -> subhalo[mass_is_instance]
        """
        if self.mass_is_model:
            return f"__{conf.instance['notation']['setup_tags']['subhalo']['mass_is_model']}"
        return f"__{conf.instance['notation']['setup_tags']['subhalo']['mass_is_instance']}"

    @property
    def source_is_model_tag(self) -> str:
        """
        Tags if the lens source model during the subhalo pipeline is model or instance.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        source_is_model = `True` -> setup[source_is_model]
        source_is_model = `False` -> subhalo[source_is_instance]
        """
        if self.source_is_model:
            return f"__{conf.instance['notation']['setup_tags']['subhalo']['source_is_model']}"
        return f"__{conf.instance['notation']['setup_tags']['subhalo']['source_is_instance']}"

    @property
    def grid_size_tag(self) -> str:
        """
        Tags the 2D dimensions of the grid (e.g. grid_size x grid_size) that the subhalo search is performed for.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - grid_size=3 -> subhalo[grid_size_3]
        - grid_size=4 -> subhalo[grid_size_4]

        Returns
        -------
        str
            The tag of the grid size.
        """
        return f"__{conf.instance['notation']['setup_tags']['subhalo']['grid_size']}_{str(self.grid_size)}"

    @property
    def subhalo_centre_tag(self) -> str:
        """
        Tags if the subhalo mass model centre of the pipeline is fixed to an input value, to customize pipeline
        output paths.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        subhalo_centre = None -> setup
        subhalo_centre = (1.0, 1.0) -> subhalo[centre_(1.0, 1.0)]
        subhalo_centre = (3.0, -2.0) -> subhalo[centre_(3.0, -2.0)]
        """
        if self.subhalo_instance is None:
            return ""
        else:
            y = "{0:.2f}".format(self.subhalo_instance.centre[0])
            x = "{0:.2f}".format(self.subhalo_instance.centre[1])
            return (
                "__"
                + conf.instance["notation"]["setup_tags"]["subhalo"]["subhalo_centre"]
                + "_("
                + y
                + ","
                + x
                + ")"
            )

    @property
    def subhalo_mass_at_200_tag(self) -> str:
        """
        Tags if the subhalo mass model mass_at_200 of the pipeline is fixed to an input value, to
        customize pipeline output paths.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        subhalo_mass_at_200 = None -> No Tag
        subhalo_mass_at_200 = 1e8 -> subhalo[mass_1.0e+08]
        subhalo_mass_at_200 = 1e9 -> subhalo[mass_1.0e+09]
        """
        if self.subhalo_instance is None:
            return ""
        else:

            return (
                "__"
                + conf.instance["notation"]["setup_tags"]["subhalo"]["mass_at_200"]
                + "_"
                + "{0:.1e}".format(self.subhalo_instance.mass_at_200)
            )


class SetupPipeline(setup.SetupPipeline):
    def __init__(
        self,
        path_prefix: str = None,
        redshift_lens: float = 0.5,
        redshift_source: float = 1.0,
        setup_hyper: setup.SetupHyper = None,
        setup_light: Union[
            setup.SetupLightParametric, setup.SetupLightInversion
        ] = None,
        setup_mass: Union[SetupMassTotal, SetupMassLightDark] = None,
        setup_source: Union[SetupSourceParametric, SetupSourceInversion] = None,
        setup_smbh: setup.SetupSMBH = None,
        setup_subhalo: SetupSubhalo = None,
    ):
        """
        The setup of a ``Pipeline``, which controls how **PyAutoLens** template pipelines runs, for example controlling
        assumptions about the bulge-disk model or the model used to fit the source galaxy.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

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
        setup_light : SetupLightParametric
            The setup of the light profile modeling (e.g. for bulge-disk models if they are geometrically aligned).
        setup_mass : SetupMassTotal or SetupMassLightDark
            The setup of the mass modeling (e.g. if a constant mass to light ratio is used).
        setup_source : SetupSourceParametric or SetupSourceInversion
            The setup of the source analysis (e.g. the `LightProfile`, `Pixelization` or `Regularization` used).
        setup_smbh : SetupSMBH
            The setup of a SMBH in the mass model, if included.
        setup_subhalo : SetupSubhalo
            The setup of a subhalo in the mass model, if included.
        """

        super().__init__(
            path_prefix=path_prefix,
            setup_hyper=setup_hyper,
            setup_light=setup_light,
            setup_mass=setup_mass,
            setup_smbh=setup_smbh,
        )

        self.setup_source = setup_source

        self.redshift_lens = redshift_lens
        self.redshift_source = redshift_source
        self.setup_subhalo = setup_subhalo

    @property
    def tag(self) -> str:
        """
        The overall pipeline tag, which customizes the 'setup' folder the results are output to.

        For the the default configuration files `config/notation/setup_tags.ini` examples of tagging are as follows:

        - setup__hyper[galaxies__bg_noise]__light[bulge_sersic__disk__exp_light_centre_(1.00,2.00)]
        - "setup__smbh[point_mass__centre_fixed]"
        """

        setup_tag = conf.instance["notation"]["setup_tags"]["pipeline"]["pipeline"]

        hyper_tag = self._pipeline_tag_from_setup(setup=self.setup_hyper)
        light_tag = self._pipeline_tag_from_setup(setup=self.setup_light)
        mass_tag = self._pipeline_tag_from_setup(setup=self.setup_mass)
        source_tag = self._pipeline_tag_from_setup(setup=self.setup_source)
        smbh_tag = self._pipeline_tag_from_setup(setup=self.setup_smbh)
        subhalo_tag = self._pipeline_tag_from_setup(setup=self.setup_subhalo)

        return f"{setup_tag}{hyper_tag}{light_tag}{mass_tag}{source_tag}{smbh_tag}{subhalo_tag}"
