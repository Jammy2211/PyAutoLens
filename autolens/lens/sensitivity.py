from typing import Optional

import autofit as af

from autofit.non_linear.grid.sensitivity.result import SensitivityResult

import autofit as af
import autoarray as aa

from autolens.lens.ray_tracing import Tracer

import autolens.plot as aplt


class SubhaloSensitivityResult(SensitivityResult):
    def __init__(
        self,
        result_sensitivity: SensitivityResult,
    ):
        """
        The results of a subhalo sensitivity mapping analysis, where dark matter halos are used to simulate many
        strong lens datasets which are fitted to quantify how detectable they are.

        Parameters
        ----------
        result_sensitivity
            The results of a sensitivity mapping analysis where.
        """

        super().__init__(
            samples=result_sensitivity.samples,
            perturb_samples=result_sensitivity.perturb_samples,
            shape=result_sensitivity.shape,
        )

    def _array_2d_from(self, values) -> aa.Array2D:
        """
        Returns an `Array2D` where the input values are reshaped from list of lists to a 2D array, which is
        suitable for plotting.

        For example, this function may return the 2D array of the increases in log evidence for every lens model
        fitted with a DM subhalo in the sensitivity mapping compared to the model without a DM subhalo.

        The orientation of the 2D array and its values are chosen to ensure that when this array is plotted, DM
        subhalos with positive y and negative x `centre` coordinates appear in the top-left of the image.

        Parameters
        ----------
        values_native
            The list of list of values which are mapped to the 2D array (e.g. the `log_evidence` difference of every
            lens model with a DM subhalo compared to the one without).

        Returns
        -------
        The 2D array of values, where the values are mapped from the input list of lists.
        """
        values_reshaped = [value for values in values.native for value in values]

        return aa.Array2D.from_yx_and_values(
            y=[centre[0] for centre in self.physical_centres_lists],
            x=[centre[1] for centre in self.physical_centres_lists],
            values=values_reshaped,
            pixel_scales=self.physical_step_sizes,
            shape_native=self.shape,
        )

    def figure_of_merit_array(
        self,
        use_log_evidences: bool = True,
        remove_zeros: bool = False,
    ) -> aa.Array2D:
        """
        Returns an `Array2D` where the values are the figure of merit (`log_evidence` or `log_likelihood` difference)
        of every lens model on the sensitivity mapping grid.

        Values below zero may be rounded to zero, to prevent the figure of merit map being dominated by low values

        Parameters
        ----------
        use_log_evidences
            If `True`, the figure of merit values are the log evidences of every lens model on the grid search.
            If `False`, they are the log likelihoods.
        remove_zeros
            If `True`, the figure of merit array is altered so that all values below 0.0 and set to 0.0. For plotting
            relative figures of merit for Bayesian model comparison, this is convenient to remove negative values
            and produce a clearer visualization of the overlay.
        """

        figures_of_merits = self.figure_of_merits(
            use_log_evidences=use_log_evidences,
        )

        if remove_zeros:
            figures_of_merits = af.GridList(
                values=[fom if fom > 0.0 else 0.0 for fom in figures_of_merits],
                shape=figures_of_merits.shape,
            )

        return self._array_2d_from(values=figures_of_merits)


class SubhaloSensitivityPlotter:
    def __init__(
        self,
        grid: aa.type.Grid2DLike,
        mask: aa.Mask2D,
        tracer_perturb: Optional[Tracer] = None,
        tracer_no_perturb: Optional[Tracer] = None,
        source_image: Optional[aa.Array2D] = None,
        result_sensitivity: Optional[SensitivityResult] = None,
        mat_plot_2d: aplt.MatPlot2D = aplt.MatPlot2D(),
        visuals_2d: aplt.Visuals2D = aplt.Visuals2D(),
        include_2d: aplt.Include2D = aplt.Include2D(),
    ):
        """
        Plots the simulated datasets and results of a sensitivity mapping analysis, where dark matter halos are used
        to simulate many strong lens datasets which are fitted to quantify how detectable they are.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `MassProfile` and plotted via the visuals object, if the corresponding entry is `True` in
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        tracer
            The tracer the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the tracer's light and mass quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 2D plots.
        """
        self.grid = grid
        self.mask = mask
        self.tracer_perturb = tracer_perturb
        self.tracer_no_perturb = tracer_no_perturb
        self.source_image = source_image
        self.result_sensitivity = result_sensitivity
        self.mat_plot_2d = mat_plot_2d
        self.visuals_2d = visuals_2d
        self.include_2d = include_2d

    def subplot_tracer_images(self):
        """
        Output the tracer images of the dataset simulated for sensitivity mapping as a .png subplot.

        This dataset corresponds to a single grid-cell on the sensitivity mapping grid and therefore will be output
        many times over the entire sensitivity mapping grid.

        The subplot includes the overall image, the lens galaxy image, the lensed source galaxy image and the source
        galaxy image interpolated to a uniform grid.

        Images are masked before visualization, so that they zoom in on the region of interest which is actually
        fitted.
        """

        grid = aa.Grid2D.from_mask(mask=self.mask)

        image = self.tracer_perturb.image_2d_from(grid=grid).binned
        lensed_source_image = self.tracer_perturb.image_2d_via_input_plane_image_from(
            grid=grid, plane_image=self.source_image
        ).binned
        lensed_source_image_no_perturb = (
            self.tracer_no_perturb.image_2d_via_input_plane_image_from(
                grid=grid, plane_image=self.source_image
            ).binned
        )

        plotter = aplt.Array2DPlotter(
            array=image,
            mat_plot_2d=self.mat_plot_2d,
        )
        plotter.open_subplot_figure(number_subplots=6)
        plotter.set_title("Image")
        plotter.figure_2d()

        grid = self.mask.derive_grid.unmasked_sub_1

        visuals_2d = aplt.Visuals2D(
            mask=self.mask,
            tangential_critical_curves=self.tracer_perturb.tangential_critical_curve_list_from(
                grid=grid
            ),
            radial_critical_curves=self.tracer_perturb.radial_critical_curve_list_from(
                grid=grid
            ),
        )

        plotter = aplt.Array2DPlotter(
            array=lensed_source_image,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
        )
        plotter.set_title("Lensed Source Image")
        plotter.figure_2d()

        visuals_2d = aplt.Visuals2D(
            mask=self.mask,
            tangential_caustics=self.tracer_perturb.tangential_caustic_list_from(
                grid=grid
            ),
            radial_caustics=self.tracer_perturb.radial_caustic_list_from(grid=grid),
        )

        plotter = aplt.Array2DPlotter(
            array=self.source_image.binned,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
        )
        plotter.set_title("Source Image")
        plotter.figure_2d()

        plotter = aplt.Array2DPlotter(
            array=self.tracer_perturb.convergence_2d_from(grid=grid).binned,
            mat_plot_2d=self.mat_plot_2d,
        )
        plotter.set_title("Convergence")
        plotter.figure_2d()

        visuals_2d = aplt.Visuals2D(
            mask=self.mask,
            tangential_critical_curves=self.tracer_no_perturb.tangential_critical_curve_list_from(
                grid=grid
            ),
            radial_critical_curves=self.tracer_no_perturb.radial_critical_curve_list_from(
                grid=grid
            ),
        )

        plotter = aplt.Array2DPlotter(
            array=lensed_source_image,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
        )
        plotter.set_title("Lensed Source Image (No Subhalo)")
        plotter.figure_2d()

        residual_map = (
            lensed_source_image.binned - lensed_source_image_no_perturb.binned
        )

        plotter = aplt.Array2DPlotter(
            array=residual_map,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=visuals_2d,
        )
        plotter.set_title("Residual Map (Subhalo - No Subhalo)")
        plotter.figure_2d()

        plotter.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_lensed_images"
        )
        plotter.close_subplot_figure()
