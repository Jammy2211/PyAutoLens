from os import path
import autofit as af
from astropy import cosmology as cosmo
from autogalaxy.galaxy import fit_galaxy
from autogalaxy.galaxy import masked_galaxy_data
from autogalaxy.pipeline.phase import abstract
from autolens.pipeline import visualizer


class Analysis(af.Analysis):
    def __init__(self, cosmology, results):
        self.cosmology = cosmology
        self.results = results
        self.visualizer = visualizer.PhaseGalaxyVisualizer()


# noinspection PyAbstractClass
class AnalysisSingle(Analysis):
    def __init__(self, galaxy_data, cosmology, results=None):
        super().__init__(cosmology=cosmology, results=results)

        self.galaxy_data = galaxy_data

    def log_likelihood_function(self, instance):
        fit = self.fit_for_instance(instance=instance)
        return fit.figure_of_merit

    def visualize(self, paths, instance, during_analysis):
        fit = self.fit_for_instance(instance=instance)

        self.visualizer.plot_galaxy_fit_subplot(fit)

        if during_analysis:
            self.visualizer.plot_fit_individuals(fit)
        else:

            if self.visualizer.plot_ray_tracing_all_at_end_png:
                self.visualizer.plot_fit_individuals(
                    fit=fit, plot_all=True, image_format="png"
                )

            if self.visualizer.plot_ray_tracing_all_at_end_fits:
                self.visualizer.plot_fit_individuals(
                    fit=fit, plot_all=True, image_format="fits", path_suffix="fits"
                )

        return fit

    def fit_for_instance(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the masked_imaging in
        this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit: Fit
            A fractional value indicating how well this model fit and the model
            masked_imaging itself
        """
        return fit_galaxy.FitGalaxy(
            masked_galaxy_dataset=self.galaxy_data, model_galaxies=instance.galaxies
        )


# noinspection PyAbstractClass
class AnalysisDeflections(Analysis):
    def __init__(self, galaxy_data_y, galaxy_data_x, cosmology, results=None):
        super().__init__(cosmology=cosmology, results=results)

        self.galaxy_data_y = galaxy_data_y
        self.galaxy_data_x = galaxy_data_x

    def log_likelihood_function(self, instance):
        fit_y, fit_x = self.fit_for_instance(instance=instance)
        return fit_y.figure_of_merit + fit_x.figure_of_merit

    def visualize(self, paths, instance, during_analysis):

        fit_y, fit_x = self.fit_for_instance(instance=instance)

        if self.visualizer.plot_subplot_galaxy_fit:
            self.visualizer.plot_galaxy_fit_subplot(fit_y, path_suffix="fit_y_")
            self.visualizer.plot_galaxy_fit_subplot(fit_x, path_suffix="fit_x_")

        if during_analysis:
            self.visualizer.plot_fit_individuals(fit_y, path_suffix="fit_y")
            self.visualizer.plot_fit_individuals(fit_x, path_suffix="fit_x")
        else:
            if self.visualizer.plot_ray_tracing_all_at_end_png:
                self.visualizer.plot_fit_individuals(
                    fit_y, path_suffix=path.join("fits", "fit_y"), plot_all=True
                )
                self.visualizer.plot_fit_individuals(
                    fit_x, path_suffix=path.join("fits", "fit_x"), plot_all=True
                )

            if self.visualizer.plot_ray_tracing_all_at_end_fits:
                self.visualizer.plot_fit_individuals(
                    fit_y,
                    path_suffix=path.join("fits", "fit_y"),
                    plot_all=True,
                    image_format="fits",
                )
                self.visualizer.plot_fit_individuals(
                    fit_x,
                    path_suffix=path.join("fits", "fit_x"),
                    plot_all=True,
                    image_format="fits",
                )

        return fit_y, fit_x

    def fit_for_instance(self, instance):

        fit_y = fit_galaxy.FitGalaxy(
            masked_galaxy_dataset=self.galaxy_data_y, model_galaxies=instance.galaxies
        )
        fit_x = fit_galaxy.FitGalaxy(
            masked_galaxy_dataset=self.galaxy_data_x, model_galaxies=instance.galaxies
        )

        return fit_y, fit_x


class PhaseGalaxy(abstract.AbstractPhase):
    galaxies = af.PhaseProperty("galaxies")

    Analysis = Analysis

    def __init__(
        self,
        name,
        search,
        folders=tuple(),
        galaxies=None,
        use_image=False,
        use_convergence=False,
        use_potential=False,
        use_deflections=False,
        sub_size=2,
        pixel_scales_interp=None,
        cosmology=cosmo.Planck15,
    ):
        """
        A phase in an lens pipeline. Uses the set non_linear search to try to fit
        models and hyper_galaxies passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        sub_size: int
            The side length of the subgrid
        """

        super(PhaseGalaxy, self).__init__(name=name, folders=folders, search=search)
        self.cosmology = cosmology
        self.use_image = use_image
        self.use_convergence = use_convergence
        self.use_potential = use_potential
        self.use_deflections = use_deflections
        self.galaxies = galaxies
        self.sub_size = sub_size
        self.pixel_scales_interp = pixel_scales_interp

    def run(self, galaxy_data, mask, info=None, pickle_files=None, results=None):
        """
        Run this phase.

        Parameters
        ----------
        galaxy_data
        mask: Mask2D
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has
            been executed

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper_galaxies.
        """
        analysis = self.make_analysis(
            galaxy_data=galaxy_data, results=results, mask=mask
        )

        self.save_metadata(galaxy_data.name)
        self.model = self.model.populate(results)

        result = self.run_analysis(
            analysis=analysis, info=info, pickle_files=pickle_files
        )

        return self.make_result(result, analysis)

    def make_analysis(self, galaxy_data, mask, results=None):
        """
        Returns an lens object. Also calls the prior passing and masked_imaging modifying
        functions to allow child classes to change the behaviour of the phase.

        Parameters
        ----------
        galaxy_data
        mask: Mask2D
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens: Analysis
            An lens object that the `NonLinearSearch` calls to determine the fit of a
             set of values
        """

        if self.use_image or self.use_convergence or self.use_potential:

            galaxy_data = masked_galaxy_data.MaskedGalaxyDataset(
                galaxy_data=galaxy_data[0],
                mask=mask,
                pixel_scales_interp=self.pixel_scales_interp,
                use_image=self.use_image,
                use_convergence=self.use_convergence,
                use_potential=self.use_potential,
                use_deflections_y=self.use_deflections,
                use_deflections_x=self.use_deflections,
            )

            return AnalysisSingle(
                galaxy_data=galaxy_data, cosmology=self.cosmology, results=results
            )

        elif self.use_deflections:

            galaxy_data_y = masked_galaxy_data.MaskedGalaxyDataset(
                galaxy_data=galaxy_data[0],
                mask=mask,
                pixel_scales_interp=self.pixel_scales_interp,
                use_image=self.use_image,
                use_convergence=self.use_convergence,
                use_potential=self.use_potential,
                use_deflections_y=self.use_deflections,
                use_deflections_x=False,
            )

            galaxy_data_x = masked_galaxy_data.MaskedGalaxyDataset(
                galaxy_data=galaxy_data[1],
                mask=mask,
                pixel_scales_interp=self.pixel_scales_interp,
                use_image=self.use_image,
                use_convergence=self.use_convergence,
                use_potential=self.use_potential,
                use_deflections_y=False,
                use_deflections_x=self.use_deflections,
            )

            return AnalysisDeflections(
                galaxy_data_y=galaxy_data_y,
                galaxy_data_x=galaxy_data_x,
                cosmology=self.cosmology,
                results=results,
            )

    # noinspection PyAbstractClass
