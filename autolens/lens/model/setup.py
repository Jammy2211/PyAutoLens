from typing import Optional

import autofit as af
import autogalaxy as ag


class SetupHyper(ag.SetupHyper):
    def __init__(
        self,
        hyper_galaxies_lens: bool = False,
        hyper_galaxies_source: bool = False,
        hyper_image_sky: Optional[type(ag.hyper_data.HyperImageSky)] = None,
        hyper_background_noise: Optional[
            type(ag.hyper_data.HyperBackgroundNoise)
        ] = None,
        hyper_fixed_after_source: bool = False,
        search_inversion_cls: Optional[af.NonLinearSearch] = None,
        search_noise_cls: Optional[af.NonLinearSearch] = None,
        search_bc_cls: Optional[af.NonLinearSearch] = None,
        search_inversion_dict: Optional[dict] = None,
        search_noise_dict: Optional[dict] = None,
        search_bc_dict: Optional[dict] = None,
    ):
        """
        The hyper setup of a pipeline, which controls how hyper-features in PyAutoLens template pipelines run,
        for example controlling whether hyper galaxies are used to scale the noise and the non-linear searches used
        in these searchs.

        Users can write their own pipelines which do not use or require the *SetupHyper* class.

        Parameters
        ----------
        hyper_galaxies
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used to scale the
            noise-map of the dataset throughout the fitting.
        hyper_image_sky 
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            image's background sky component in the model.
        hyper_background_noise
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            noise-map's background component in the model.
        hyper_fixed_after_source
            If `True`, the hyper parameters are fixed and not updated after a desnated pipeline in the analysis. For
            the `SLaM` pipelines this is after the `SourcePipeline`. This allow Bayesian model comparison to be
            performed objected between later searchs in a pipeline.
        search_inversion_cls
            The non-linear search used by every hyper model-fit search.
        search_inversion_dict
            The dictionary of search options for the hyper model-fit searches.
        """
        hyper_galaxies = hyper_galaxies_lens or hyper_galaxies_source

        super().__init__(
            hyper_galaxies=hyper_galaxies,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            search_inversion_cls=search_inversion_cls,
            search_noise_cls=search_noise_cls,
            search_bc_cls=search_bc_cls,
            search_inversion_dict=search_inversion_dict,
            search_noise_dict=search_noise_dict,
            search_bc_dict=search_bc_dict,
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

    def hyper_galaxy_lens_from(self, result: af.Result, noise_factor_is_model=False):
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
        index
            The index of the previous search the `HyperGalaxy` `Model` is passed from.
        noise_factor_is_model
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
            return self.hyper_galaxy_via_galaxy_model_from(
                galaxy_model=result.hyper.model.galaxies.lens,
                galaxy_instance=result.hyper.instance.galaxies.lens,
                noise_factor_is_model=noise_factor_is_model,
            )

        return self.hyper_galaxy_via_galaxy_model_from(
            galaxy_model=result.model.galaxies.lens,
            galaxy_instance=result.instance.galaxies.lens,
            noise_factor_is_model=noise_factor_is_model,
        )

    def hyper_galaxy_source_from(self, result: af.Result, noise_factor_is_model=False):
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
        index
            The index of the previous search the `HyperGalaxy` `Model` is passed from.
        noise_factor_is_model
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
            return self.hyper_galaxy_via_galaxy_model_from(
                galaxy_model=result.hyper.model.galaxies.source,
                galaxy_instance=result.hyper.instance.galaxies.source,
                noise_factor_is_model=noise_factor_is_model,
            )

        return self.hyper_galaxy_via_galaxy_model_from(
            galaxy_model=result.model.galaxies.source,
            galaxy_instance=result.instance.galaxies.source,
            noise_factor_is_model=noise_factor_is_model,
        )

    def hyper_galaxy_via_galaxy_model_from(
        self, galaxy_model, galaxy_instance, noise_factor_is_model=False
    ):

        hyper_galaxy = af.Model(ag.HyperGalaxy)

        if galaxy_model.hyper_galaxy is None:
            return None

        if not noise_factor_is_model:
            hyper_galaxy.noise_factor = galaxy_instance.hyper_galaxy.noise_factor
        else:
            hyper_galaxy.noise_factor = af.LogUniformPrior(
                lower_limit=1e-4,
                upper_limit=2.0 * galaxy_instance.hyper_galaxy.noise_factor,
            )

        hyper_galaxy.contribution_factor = (
            galaxy_instance.hyper_galaxy.contribution_factor
        )
        hyper_galaxy.noise_power = galaxy_instance.hyper_galaxy.noise_power

        return hyper_galaxy
