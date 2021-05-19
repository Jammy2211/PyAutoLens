from os import path
import numpy as np

import autofit as af
import autolens as al
from autolens import exc
import pytest
from autolens.analysis import result as res
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestAnalysisAbstract:

    pass


class TestAnalysisDataset:
    def test__use_border__determines_if_border_pixel_relocation_is_used(
        self, masked_imaging_7x7
    ):

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(
                    redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=100.0)
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    pixelization=al.pix.Rectangular(shape=(3, 3)),
                    regularization=al.reg.Constant(coefficient=1.0),
                ),
            )
        )

        masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
            settings=al.SettingsImaging(sub_size_inversion=2)
        )

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_pixelization=al.SettingsPixelization(use_border=True),
        )

        analysis.dataset.grid_inversion[4] = np.array([[500.0, 0.0]])

        instance = model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.fit_imaging_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.source_grid_slim[4][0] == pytest.approx(
            97.19584, 1.0e-2
        )
        assert fit.inversion.mapper.source_grid_slim[4][1] == pytest.approx(
            -3.699999, 1.0e-2
        )

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_pixelization=al.SettingsPixelization(use_border=False),
        )

        analysis.dataset.grid_inversion[4] = np.array([300.0, 0.0])

        instance = model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.fit_imaging_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.source_grid_slim[4][0] == pytest.approx(
            200.0, 1.0e-4
        )

    def test__analysis_no_positions__removes_positions_and_threshold(
        self, masked_imaging_7x7
    ):

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]),
            settings_lens=al.SettingsLens(positions_threshold=0.01),
        )

        assert analysis.no_positions.positions == None
        assert analysis.no_positions.settings_lens.positions_threshold == None


class TestAnalysisImaging:
    def test__make_result__result_imaging_is_returned(self, masked_imaging_7x7):

        model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

        search = mock.MockSearch(name="test_search")

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultImaging)

    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, masked_imaging_7x7
    ):

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal()),
                source=al.Galaxy(redshift=1.0),
            )
        )

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]),
            settings_lens=al.SettingsLens(positions_threshold=0.01),
        )

        instance = model.instance_from_unit_vector([])

        with pytest.raises(exc.RayTracingException):
            analysis.log_likelihood_function(instance=instance)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, masked_imaging_7x7
    ):
        lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))

        model = af.Collection(galaxies=af.Collection(lens=lens_galaxy))

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)
        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.FitImaging(imaging=masked_imaging_7x7, tracer=tracer)

        assert fit.log_likelihood == analysis_log_likelihood

    def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, masked_imaging_7x7
    ):

        hyper_image_sky = al.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))

        model = af.Collection(
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            galaxies=af.Collection(lens=lens_galaxy),
        )

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)
        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)
        fit = al.FitImaging(
            imaging=masked_imaging_7x7,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == analysis_log_likelihood

    def test__uses_hyper_fit_correctly(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5, light=al.lp.EllSersic(intensity=1.0), mass=al.mp.SphIsothermal
        )
        galaxies.source = al.Galaxy(redshift=1.0, light=al.lp.EllSersic())

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        lens_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        lens_hyper_image[4] = 10.0
        hyper_model_image = al.Array2D.full(
            fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {("galaxies", "lens"): lens_hyper_image}

        result = mock.MockResult(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, hyper_result=result)

        hyper_galaxy = al.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        )

        instance.galaxies.lens.hyper_galaxy = hyper_galaxy

        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        g0 = al.Galaxy(
            redshift=0.5,
            light_profile=instance.galaxies.lens.light,
            mass_profile=instance.galaxies.lens.mass,
            hyper_galaxy=hyper_galaxy,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_image=lens_hyper_image,
            hyper_minimum_value=0.0,
        )
        g1 = al.Galaxy(redshift=1.0, light_profile=instance.galaxies.source.light)

        tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

        fit = al.FitImaging(imaging=masked_imaging_7x7, tracer=tracer)

        assert (fit.tracer.galaxies[0].hyper_galaxy_image == lens_hyper_image).all()
        assert analysis_log_likelihood == fit.log_likelihood

    def test__sets_up_hyper_galaxy_images__from_results(self, masked_imaging_7x7):

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): al.Array2D.ones(
                shape_native=(3, 3), pixel_scales=1.0
            ),
            ("galaxies", "source"): al.Array2D.full(
                fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
            ),
        }

        result = mock.MockResult(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=al.Array2D.full(
                fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
            ),
        )

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, hyper_result=result)

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "lens")].native
            == np.ones((3, 3))
        ).all()

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "source")].native
            == 2.0 * np.ones((3, 3))
        ).all()

        assert (analysis.hyper_model_image.native == 3.0 * np.ones((3, 3))).all()

    def test__stochastic_log_evidences_for_instance(self, masked_imaging_7x7):

        lens_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        lens_hyper_image[4] = 10.0
        source_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        source_hyper_image[4] = 10.0
        hyper_model_image = al.Array2D.full(
            fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): lens_hyper_image,
            ("galaxies", "source"): source_hyper_image,
        }

        result = mock.MockResult(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=1.0)
        )
        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiMagnification(shape=(3, 3)),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, hyper_result=result)

        stochastic_log_evidences = analysis.stochastic_log_evidences_for_instance(
            instance=instance
        )

        assert stochastic_log_evidences is None

        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(pixels=9),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7, hyper_result=result)

        stochastic_log_evidences = analysis.stochastic_log_evidences_for_instance(
            instance=instance
        )

        assert stochastic_log_evidences[0] != stochastic_log_evidences[1]


class TestAnalysisInterferometer:
    def test__make_result__result_interferometer_is_returned(self, interferometer_7):

        model = af.Collection(galaxies=af.Collection(galaxy_0=al.Galaxy(redshift=0.5)))

        search = mock.MockSearch(name="test_search")

        analysis = al.AnalysisInterferometer(dataset=interferometer_7)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultInterferometer)

    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, interferometer_7, mask_2d_7x7
    ):

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal()),
                source=al.Galaxy(redshift=1.0),
            )
        )

        analysis = al.AnalysisInterferometer(
            dataset=interferometer_7,
            positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]),
            settings_lens=al.SettingsLens(positions_threshold=0.01),
        )

        instance = model.instance_from_unit_vector([])

        with pytest.raises(exc.RayTracingException):
            analysis.log_likelihood_function(instance=instance)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, interferometer_7
    ):
        lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))

        model = af.Collection(galaxies=af.Collection(lens=lens_galaxy))

        analysis = al.AnalysisInterferometer(dataset=interferometer_7)

        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.FitInterferometer(interferometer=interferometer_7, tracer=tracer)

        assert fit.log_likelihood == analysis_log_likelihood

    def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, interferometer_7
    ):
        hyper_background_noise = al.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))

        model = af.Collection(
            galaxies=af.Collection(lens=lens_galaxy),
            hyper_background_noise=hyper_background_noise,
        )

        analysis = al.AnalysisInterferometer(dataset=interferometer_7)

        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit = al.FitInterferometer(
            interferometer=interferometer_7,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == analysis_log_likelihood

    def test__sets_up_hyper_galaxy_visibiltiies__from_results(self, interferometer_7):

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): al.Array2D.ones(
                shape_native=(3, 3), pixel_scales=1.0
            ),
            ("galaxies", "source"): al.Array2D.full(
                fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
            ),
        }

        hyper_galaxy_visibilities_path_dict = {
            ("galaxies", "lens"): al.Visibilities.full(fill_value=4.0, shape_slim=(7,)),
            ("galaxies", "source"): al.Visibilities.full(
                fill_value=5.0, shape_slim=(7,)
            ),
        }

        result = mock.MockResult(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=al.Array2D.full(
                fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
            ),
            hyper_galaxy_visibilities_path_dict=hyper_galaxy_visibilities_path_dict,
            hyper_model_visibilities=al.Visibilities.full(
                fill_value=6.0, shape_slim=(7,)
            ),
        )

        analysis = al.AnalysisInterferometer(
            dataset=interferometer_7, hyper_result=result
        )

        analysis.set_hyper_dataset(result=result)

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "lens")].native
            == np.ones((3, 3))
        ).all()

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "source")].native
            == 2.0 * np.ones((3, 3))
        ).all()

        assert (analysis.hyper_model_image.native == 3.0 * np.ones((3, 3))).all()

        assert (
            analysis.hyper_galaxy_visibilities_path_dict[("galaxies", "lens")]
            == (4.0 + 4.0j) * np.ones((7,))
        ).all()

        assert (
            analysis.hyper_galaxy_visibilities_path_dict[("galaxies", "source")]
            == (5.0 + 5.0j) * np.ones((7,))
        ).all()

        assert (analysis.hyper_model_visibilities == (6.0 + 6.0j) * np.ones((7,))).all()

    def test__stochastic_log_evidences_for_instance(self, interferometer_7):

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=1.2)
        )
        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(pixels=5),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        lens_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        lens_hyper_image[4] = 10.0
        source_hyper_image = al.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        source_hyper_image[4] = 10.0
        hyper_model_image = al.Array2D.full(
            fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): lens_hyper_image,
            ("galaxies", "source"): source_hyper_image,
        }

        result = mock.MockResult(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = al.AnalysisInterferometer(
            dataset=interferometer_7,
            settings_lens=al.SettingsLens(stochastic_samples=2),
            hyper_result=result,
        )

        log_evidences = analysis.stochastic_log_evidences_for_instance(
            instance=instance
        )

        assert len(log_evidences) == 2
        assert log_evidences[0] != log_evidences[1]


class TestAnalysisPoint:
    def test__make_result__result_imaging_is_returned(self, point_dict):

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
            )
        )

        search = mock.MockSearch(name="test_search")

        solver = mock.MockPositionsSolver(
            model_positions=point_dict["point_0"].positions
        )

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultPoint)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, positions_x2, positions_x2_noise_map
    ):

        point_dataset = al.PointDataset(
            name="point_0",
            positions=positions_x2,
            positions_noise_map=positions_x2_noise_map,
        )

        point_dict = al.PointDict(point_dataset_list=[point_dataset])

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
            )
        )

        solver = mock.MockPositionsSolver(model_positions=positions_x2)

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit_positions = al.FitPositionsImage(
            name="point_0",
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=solver,
        )

        assert fit_positions.chi_squared == 0.0
        assert fit_positions.log_likelihood == analysis_log_likelihood

        model_positions = al.Grid2DIrregular([(0.0, 1.0), (1.0, 2.0)])
        solver = mock.MockPositionsSolver(model_positions=model_positions)

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        fit_positions = al.FitPositionsImage(
            name="point_0",
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=solver,
        )

        assert fit_positions.residual_map.in_list == [1.0, 1.0]
        assert fit_positions.chi_squared == 2.0
        assert fit_positions.log_likelihood == analysis_log_likelihood

    def test__figure_of_merit__includes_fit_fluxes(
        self, positions_x2, positions_x2_noise_map, fluxes_x2, fluxes_x2_noise_map
    ):

        point_dataset = al.PointDataset(
            name="point_0",
            positions=positions_x2,
            positions_noise_map=positions_x2_noise_map,
            fluxes=fluxes_x2,
            fluxes_noise_map=fluxes_x2_noise_map,
        )

        point_dict = al.PointDict(point_dataset_list=[point_dataset])

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(
                    redshift=0.5,
                    sis=al.mp.SphIsothermal(einstein_radius=1.0),
                    point_0=al.ps.PointFlux(flux=1.0),
                )
            )
        )

        solver = mock.MockPositionsSolver(model_positions=positions_x2)

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        instance = model.instance_from_unit_vector([])

        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit_positions = al.FitPositionsImage(
            name="point_0",
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=solver,
        )

        fit_fluxes = al.FitFluxes(
            name="point_0",
            fluxes=fluxes_x2,
            noise_map=fluxes_x2_noise_map,
            positions=positions_x2,
            tracer=tracer,
        )

        assert (
            fit_positions.log_likelihood + fit_fluxes.log_likelihood
            == analysis_log_likelihood
        )

        model_positions = al.Grid2DIrregular([(0.0, 1.0), (1.0, 2.0)])
        solver = mock.MockPositionsSolver(model_positions=model_positions)

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        fit_positions = al.FitPositionsImage(
            name="point_0",
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=solver,
        )

        fit_fluxes = al.FitFluxes(
            name="point_0",
            fluxes=fluxes_x2,
            noise_map=fluxes_x2_noise_map,
            positions=positions_x2,
            tracer=tracer,
        )

        assert fit_positions.residual_map.in_list == [1.0, 1.0]
        assert fit_positions.chi_squared == 2.0
        assert (
            fit_positions.log_likelihood + fit_fluxes.log_likelihood
            == analysis_log_likelihood
        )
