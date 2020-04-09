from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autolens as al
from autolens import exc
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestPhase:
    def test__extend_with_hyper_and_pixelizations(self):

        phase_no_pixelization = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO, phase_name="test_phase"
        )

        phase_extended = phase_no_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=False
        )
        assert phase_extended == phase_no_pixelization

        # This phase does not have a pixelization, so even though inversion=True it will not be extended

        phase_extended = phase_no_pixelization.extend_with_multiple_hyper_phases(
            inversion=True
        )
        assert phase_extended == phase_no_pixelization

        phase_with_pixelization = al.PhaseImaging(
            galaxies=dict(
                source=al.GalaxyModel(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular,
                    regularization=al.reg.Constant,
                )
            ),
            non_linear_class=mock_pipeline.MockNLO,
            phase_name="test_phase",
        )

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=False
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase
        assert type(phase_extended.hyper_phases[1]) == al.HyperGalaxyPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True, hyper_galaxy_phase_first=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase
        assert type(phase_extended.hyper_phases[1]) == al.InversionPhase


class TestMakeAnalysis:
    def test__mask_input_uses_mask(self, phase_imaging_7x7, imaging_7x7):
        # If an input mask is supplied we use mask input.

        mask_input = al.Mask.circular(
            shape_2d=imaging_7x7.shape_2d, pixel_scales=1.0, sub_size=1, radius=1.5
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_input, results=mock_pipeline.MockResults()
        )

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.pixel_scales == mask_input.pixel_scales

    def test__mask_changes_sub_size_depending_on_phase_attribute(
        self, phase_imaging_7x7, imaging_7x7
    ):
        # If an input mask is supplied we use mask input.

        mask_input = al.Mask.circular(
            shape_2d=imaging_7x7.shape_2d, pixel_scales=1, sub_size=1, radius=1.5
        )

        phase_imaging_7x7.meta_dataset.sub_size = 1
        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_input, results=mock_pipeline.MockResults()
        )

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.sub_size == 1
        assert analysis.masked_imaging.mask.pixel_scales == mask_input.pixel_scales

        phase_imaging_7x7.meta_dataset.sub_size = 2
        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_input, results=mock_pipeline.MockResults()
        )

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.sub_size == 2
        assert analysis.masked_imaging.mask.pixel_scales == mask_input.pixel_scales

    def test__positions_are_input__are_used_in_analysis(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        # If position threshold is input (not None) and positions are input, make the positions part of the lens dataset.

        phase_imaging_7x7.meta_dataset.positions_threshold = 0.2

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=[[(1.0, 1.0), (2.0, 2.0)]],
            results=mock_pipeline.MockResults(),
        )

        assert (analysis.masked_dataset.positions[0][0] == np.array([1.0, 1.0])).all()
        assert (analysis.masked_dataset.positions[0][1] == np.array([2.0, 2.0])).all()
        assert analysis.masked_imaging.positions_threshold == 0.2

        # If position threshold is input (not None) and but no positions are supplied, raise an error

        with pytest.raises(exc.PhaseException):

            phase_imaging_7x7.meta_dataset.positions_threshold = 0.2

            phase_imaging_7x7.make_analysis(
                dataset=imaging_7x7,
                mask=mask_7x7,
                positions=None,
                results=mock_pipeline.MockResults(),
            )

            phase_imaging_7x7.meta_dataset.positions_threshold = 0.2

            phase_imaging_7x7.make_analysis(
                dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
            )

    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(source=al.Galaxy(redshift=0.5)),
            positions_threshold=50.0,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=[[(1.0, 1.0), (2.0, 2.0)]],
            results=mock_pipeline.MockResults(),
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(source=al.Galaxy(redshift=0.5)),
            positions_threshold=0.0,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=[[(1.0, 1.0), (2.0, 2.0)]],
            results=mock_pipeline.MockResults(),
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.RayTracingException):
            analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer
            )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(source=al.Galaxy(redshift=0.5)),
            positions_threshold=0.5,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=[[(1.0, 0.0), (-1.0, 0.0)]],
            results=mock_pipeline.MockResults(),
        )
        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.0)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=0.0)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        with pytest.raises(exc.RayTracingException):
            analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer
            )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]],
            results=mock_pipeline.MockResults(),
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=[[(0.0, 0.0), (0.0, 0.0)], [(100.0, 0.0), (0.0, 0.0)]],
            results=mock_pipeline.MockResults(),
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.RayTracingException):
            analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer
            )

    def test__inversion_resolution_error_raised_if_above_inversion_pixel_limit(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                source=al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular(shape=(3, 3)),
                    regularization=al.reg.Constant(),
                )
            ),
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.masked_dataset.check_inversion_pixels_are_below_limit_via_tracer(
            tracer=tracer
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                source=al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular(shape=(4, 4)),
                    regularization=al.reg.Constant(),
                )
            ),
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.PixelizationException):
            analysis.masked_dataset.check_inversion_pixels_are_below_limit_via_tracer(
                tracer=tracer
            )
            analysis.fit(instance=instance)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                source=al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular(shape=(3, 3)),
                    regularization=al.reg.Constant(),
                )
            ),
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.masked_dataset.check_inversion_pixels_are_below_limit_via_tracer(
            tracer=tracer
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                source=al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular(shape=(4, 4)),
                    regularization=al.reg.Constant(),
                )
            ),
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.PixelizationException):
            analysis.masked_dataset.check_inversion_pixels_are_below_limit_via_tracer(
                tracer=tracer
            )
            analysis.fit(instance=instance)

    def test__pixel_scale_interpolation_grid_is_input__interp_grid_used_in_analysis(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        # If use positions is true and positions are input, make the positions part of the lens dataset.

        phase_imaging_7x7.meta_dataset.pixel_scale_interpolation_grid = 0.1

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        assert analysis.masked_imaging.pixel_scale_interpolation_grid == 0.1
        assert hasattr(analysis.masked_imaging.grid, "interpolator")
        assert hasattr(analysis.masked_imaging.blurring_grid, "interpolator")

    def test__determines_if_pixelization_is_same_as_previous_phase(
        self, imaging_7x7, mask_7x7
    ):

        results = mock_pipeline.MockResults()

        results.last.hyper_combined.preload_sparse_grids_of_planes = 1

        phase_imaging_7x7 = al.PhaseImaging(phase_name="test_phase")

        results.last.pixelization = None

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.preload_sparse_grids_of_planes is None

        phase_imaging_7x7 = al.PhaseImaging(phase_name="test_phase")

        results.last.pixelization = al.pix.Rectangular

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.preload_sparse_grids_of_planes is None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular,
                    regularization=al.reg.Constant,
                )
            ],
        )

        results.last.pixelization = None

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.preload_sparse_grids_of_planes is None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular,
                    regularization=al.reg.Constant,
                )
            ],
        )

        results.last.pixelization = al.pix.Rectangular

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert analysis.masked_dataset.preload_sparse_grids_of_planes == 1


class TestPhasePickle:

    # noinspection PyTypeChecker
    def test_assertion_failure(self, imaging_7x7, mask_7x7):
        def make_analysis(*args, **kwargs):
            return mock_pipeline.GalaxiesMockAnalysis(1, 1)

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=None, positions=None
        )
        assert result is not None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=None, positions=None
        )
        assert result is not None

        class CustomPhase(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies.lens.light = al.lp.EllipticalLightProfile()

        phase_imaging_7x7 = CustomPhase(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )
        phase_imaging_7x7.make_analysis = make_analysis

        # with pytest.raises(af.exc.PipelineException):
        #     phase_imaging_7x7.run(data_type=imaging_7x7, results=None, mask=None, positions=None)
