import autofit as af
from autoconf import conf
from autogalaxy.analysis import setup
from autoarray.inversion import pixelizations as pix, regularization as reg
from autogalaxy.profiles import (
    light_profiles as lp,
    mass_profiles as mp,
    light_and_mass_profiles as lmp,
)
from autogalaxy.hyper import hyper_data as hd
from autogalaxy.galaxy import galaxy as g

from typing import Tuple, Union, Optional


class SetupHyper(setup.SetupHyper):
    def __init__(
        self,
        hyper_galaxies_lens: bool = False,
        hyper_galaxies_source: bool = False,
        hyper_image_sky: Optional[type(hd.HyperImageSky)] = None,
        hyper_background_noise: Optional[type(hd.HyperBackgroundNoise)] = None,
        hyper_fixed_after_source: bool = False,
        hyper_search: af.NonLinearSearch = None,
        evidence_tolerance: float = None,
    ):
        """
        The hyper setup of a pipeline, which controls how hyper-features in PyAutoLens template pipelines run,
        for example controlling whether hyper galaxies are used to scale the noise and the non-linear searches used
        in these phases.

        Users can write their own pipelines which do not use or require the *SetupHyper* class.

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
            If `True`, the hyper parameters are fixed and not updated after a desnated pipeline in the analysis. For
            the `SLaM` pipelines this is after the `SourcePipeline`. This allow Bayesian model comparison to be
            performed objected between later phases in a pipeline.
        hyper_search_no_inversion : af.NonLinearSearch or None
            The `NonLinearSearch` used by every inversion phase.
        hyper_search_with_inversion : af.NonLinearSearch or None
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
            hyper_search=hyper_search,
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

    def hyper_galaxy_lens_from_result(
        self, result: af.Result, noise_factor_is_model=False
    ):
        """
        Returns the `HyperGalaxy` `PriorModel` from a previous pipeline or phase of the lens galaxy in a template
        PyAutoLens pipeline.

        The `HyperGalaxy` is extracted from the `hyper` phase of the previous pipeline, and by default has its
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

        if not self.hyper_galaxies_lens:
            return None

        if hasattr(result, "hyper"):
            return self.hyper_galaxy_from_galaxy_model_and_instance(
                galaxy_model=result.hyper.model.galaxies.lens,
                galaxy_instance=result.hyper.instance.galaxies.lens,
                noise_factor_is_model=noise_factor_is_model,
            )

        return self.hyper_galaxy_from_galaxy_model_and_instance(
            galaxy_model=result.model.galaxies.lens,
            galaxy_instance=result.instance.galaxies.lens,
        )

    def hyper_galaxy_source_from_result(
        self, result: af.Result, noise_factor_is_model=False
    ):
        """
        Returns the `HyperGalaxy` `PriorModel` from a previous pipeline or phase of the source galaxy in a template
        PyAutosource pipeline.

        The `HyperGalaxy` is extracted from the `hyper` phase of the previous pipeline, and by default has its
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

        if not self.hyper_galaxies_source:
            return None

        if hasattr(result, "hyper"):
            return self.hyper_galaxy_from_galaxy_model_and_instance(
                galaxy_model=result.hyper.model.galaxies.source,
                galaxy_instance=result.hyper.instance.galaxies.source,
                noise_factor_is_model=noise_factor_is_model,
            )

        return self.hyper_galaxy_from_galaxy_model_and_instance(
            galaxy_model=result.model.galaxies.source,
            galaxy_instance=result.instance.galaxies.source,
        )

    def hyper_galaxy_from_galaxy_model_and_instance(
        self, galaxy_model, galaxy_instance, noise_factor_is_model=False
    ):

        hyper_galaxy = af.PriorModel(g.HyperGalaxy)

        if galaxy_model.hyper_galaxy is None:
            return None

        if not noise_factor_is_model:

            hyper_galaxy.noise_factor = galaxy_instance.hyper_galaxy.noise_factor

        hyper_galaxy.contribution_factor = (
            galaxy_instance.hyper_galaxy.contribution_factor
        )
        hyper_galaxy.noise_power = galaxy_instance.hyper_galaxy.noise_power

        return hyper_galaxy


class AbstractSetupMass:

    with_shear = None

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


class SetupMassLightDark(setup.SetupMassLightDark, AbstractSetupMass):
    def __init__(
        self,
        with_shear=True,
        bulge_prior_model: af.PriorModel(lmp.LightMassProfile) = lmp.EllipticalSersic,
        disk_prior_model: af.PriorModel(
            lmp.LightMassProfile
        ) = lmp.EllipticalExponential,
        envelope_prior_model: af.PriorModel(lmp.LightMassProfile) = None,
        dark_prior_model: af.PriorModel(mp.MassProfile) = mp.EllipticalNFWMCRLudlow,
        mass_centre: (float, float) = None,
        constant_mass_to_light_ratio: bool = False,
        align_bulge_dark_centre: bool = False,
    ):
        """
        The setup of the mass modeling in a pipeline for `MassProfile`'s representing the decomposed light and dark
        mass distributions, which controls how PyAutoGalaxy template pipelines run, for example controlling assumptions
        about the bulge-disk model.

        Users can write their own pipelines which do not use or require the `SetupMassLightDark` class.

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
            dark_prior_model=dark_prior_model,
            mass_centre=mass_centre,
            constant_mass_to_light_ratio=constant_mass_to_light_ratio,
            align_bulge_dark_centre=align_bulge_dark_centre,
        )

        self.with_shear = with_shear


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


class SetupSubhalo(setup.AbstractSetup):
    def __init__(
        self,
        subhalo_prior_model: af.PriorModel(mp.MassProfile) = mp.SphericalNFWMCRLudlow,
        subhalo_search: af.NonLinearSearch = None,
        source_is_model: bool = True,
        grid_dimension_arcsec: float = 3.0,
        number_of_steps: Union[Tuple[int], int] = 5,
        number_of_cores: int = 1,
        subhalo_instance=None,
    ):
        """
        The setup of a subhalo pipeline, which controls how PyAutoLens template pipelines runs.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

        Parameters
        ----------
        subhalo_search : af.NonLinearSearch
            The search used to sample parameter space in the subhalo pipeline.
        source_is_model : bool
            If `True`, the source is included as a model in the fit (for both `LightProfile` or `Inversion` sources).
            If `False` its parameters are fixed to those inferred in a previous pipeline.
        number_of_steps : int
            The 2D dimensions of the grid (e.g. number_of_steps x number_of_steps) that the subhalo search is performed for.
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
        self.number_of_steps = number_of_steps
        self.grid_dimensions_arcsec = grid_dimension_arcsec
        self.number_of_cores = number_of_cores
        self.subhalo_instance = subhalo_instance


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

        Parameters
        ----------
        path_prefix : str or None
            The prefix of folders between the output path and the search folder.
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
