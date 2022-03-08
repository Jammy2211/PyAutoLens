from os import path

import autofit as af
import autolens as al

from autolens.quantity.model.result import ResultQuantity

directory = path.dirname(path.realpath(__file__))


class TestAnalysisQuantity:
    def test__make_result__result_quantity_is_returned(
        self, dataset_quantity_7x7_array_2d
    ):

        model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

        analysis = al.AnalysisQuantity(
            dataset=dataset_quantity_7x7_array_2d, func_str="convergence_2d_from"
        )

        search = al.m.MockSearch(name="test_search")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, ResultQuantity)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, dataset_quantity_7x7_array_2d
    ):
        galaxy = al.Galaxy(redshift=0.5, light=al.mp.EllIsothermal(einstein_radius=1.0))

        model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

        analysis = al.AnalysisQuantity(
            dataset=dataset_quantity_7x7_array_2d, func_str="convergence_2d_from"
        )

        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_via_instance_from(instance=instance)

        fit = al.FitQuantity(
            dataset=dataset_quantity_7x7_array_2d,
            tracer=tracer,
            func_str="convergence_2d_from",
        )

        assert fit.log_likelihood == fit_figure_of_merit

        fit = al.FitQuantity(
            dataset=dataset_quantity_7x7_array_2d,
            tracer=tracer,
            func_str="potential_2d_from",
        )

        assert fit.log_likelihood != fit_figure_of_merit
