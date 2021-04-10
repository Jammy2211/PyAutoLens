import autofit as af
from autogalaxy.analysis import setup
from autogalaxy.profiles import mass_profiles as mp, light_and_mass_profiles as lmp
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
        search: af.NonLinearSearch = None,
        dlogz: float = None,
    ):
        """
        The hyper setup of a pipeline, which controls how hyper-features in PyAutoLens template pipelines run,
        for example controlling whether hyper galaxies are used to scale the noise and the non-linear searches used
        in these searchs.

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
            performed objected between later searchs in a pipeline.
        search : af.NonLinearSearch or None
            The non-linear search used by every hyper model-fit search.
        dlogz : float
            The evidence tolerance of the non-linear searches used in the hyper searchs, whereby higher values will
            lead them to end earlier at the expense of accuracy.
        """
        hyper_galaxies = hyper_galaxies_lens or hyper_galaxies_source

        super().__init__(
            hyper_galaxies=hyper_galaxies,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            search=search,
            dlogz=dlogz,
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
        Returns the `HyperGalaxy` `Model` from a previous pipeline or search of the lens galaxy in a template
        PyAutoLens pipeline.

        The `HyperGalaxy` is extracted from the `hyper` search of the previous pipeline, and by default has its
        parameters passed as instance's which are fixed in the next search.

        If `noise_factor_is_model` is `True` the `noise_factor` parameter of the `HyperGalaxy` is passed as a model and
        fitted for by the search. This is typically used when the lens model complexity is updated and it is possible
        that the noise-scaling performed in the previous search (using a simpler lens light model) over-scales the
        noise for the new more complex light profile.

        Parameters
        ----------
        index : int
            The index of the previous search the `HyperGalaxy` `Model` is passed from.
        noise_factor_is_model : bool
            If `True` the `noise_factor` of the `HyperGalaxy` is passed as a `model`, else it is passed as an
            `instance`.

        Returns
        -------
        af.Model(g.HyperGalaxy)
            The hyper-galaxy that is passed to the next search.
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
        Returns the `HyperGalaxy` `Model` from a previous pipeline or search of the source galaxy in a template
        PyAutosource pipeline.

        The `HyperGalaxy` is extracted from the `hyper` search of the previous pipeline, and by default has its
        parameters passed as instance's which are fixed in the next search.

        If `noise_factor_is_model` is `True` the `noise_factor` parameter of the `HyperGalaxy` is passed as a model and
        fitted for by the search. This is typically used when the source model complexity is updated and it is possible
        that the noise-scaling performed in the previous search (using a simpler source light model) over-scales the
        noise for the new more complex light profile.

        Parameters
        ----------
        index : int
            The index of the previous search the `HyperGalaxy` `Model` is passed from.
        noise_factor_is_model : bool
            If `True` the `noise_factor` of the `HyperGalaxy` is passed as a `model`, else it is passed as an
            `instance`.

        Returns
        -------
        af.Model(g.HyperGalaxy)
            The hyper-galaxy that is passed to the next search.
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

        hyper_galaxy = af.Model(g.HyperGalaxy)

        if galaxy_model.hyper_galaxy is None:
            return None

        if not noise_factor_is_model:
            hyper_galaxy.noise_factor = galaxy_instance.hyper_galaxy.noise_factor

        hyper_galaxy.contribution_factor = (
            galaxy_instance.hyper_galaxy.contribution_factor
        )
        hyper_galaxy.noise_power = galaxy_instance.hyper_galaxy.noise_power

        return hyper_galaxy
