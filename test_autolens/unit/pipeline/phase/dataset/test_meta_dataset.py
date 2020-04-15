from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autolens as al
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestSetup:
    def test__auto_positions_update__updates_correct_using_factor(
        self,
        phase_imaging_7x7,
        phase_interferometer_7,
        imaging_7x7,
        interferometer_7,
        mask_7x7,
    ):

        # Auto positioning is OFF, so use input positions + threshold.

        phase_imaging_7x7.meta_dataset.positions_threshold = 0.1
        phase_imaging_7x7.meta_dataset.auto_positions_factor = None

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=al.Coordinates(coordinates=[[(1.0, 1.0)]]),
            results=mock_pipeline.MockResults(),
        )

        assert analysis.masked_dataset.positions.in_list == [[(1.0, 1.0)]]

        # Auto positioning is ON, but there are no previous results, so use input positions.

        phase_imaging_7x7.meta_dataset.positions_threshold = 0.2
        phase_imaging_7x7.meta_dataset.auto_positions_factor = 1.0

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=al.Coordinates(coordinates=[[(1.0, 1.0)]]),
            results=mock_pipeline.MockResults(),
        )

        assert analysis.masked_dataset.positions.in_list == [[(1.0, 1.0)]]

        # Auto positioning is ON, there are previous results so use their new positions and threshold (which is
        # multiplied by the auto_positions_factor). However, only one set of positions is computed from the previous
        # result, to use input positions.

        phase_imaging_7x7.meta_dataset.positions_threshold = 0.2
        phase_imaging_7x7.meta_dataset.auto_positions_factor = 2.0

        results = mock_pipeline.MockResults(
            updated_positions=al.Coordinates(coordinates=[[(2.0, 2.0)]]),
            updated_positions_threshold=0.3,
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=al.Coordinates(coordinates=[[(1.0, 1.0)]]),
            results=results,
        )

        assert analysis.masked_dataset.positions.in_list == [[(1.0, 1.0)]]

        # Auto positioning is ON, there are previous results so use their new positions and threshold (which is
        # multiplied by the auto_positions_factor). Multiple positions are available so these are now used.

        phase_imaging_7x7.meta_dataset.positions_threshold = 0.2
        phase_imaging_7x7.meta_dataset.auto_positions_factor = 2.0

        results = mock_pipeline.MockResults(
            updated_positions=al.Coordinates(coordinates=[[(2.0, 2.0), (3.0, 3.0)]]),
            updated_positions_threshold=0.3,
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=al.Coordinates(coordinates=[[(1.0, 1.0)]]),
            results=results,
        )

        assert analysis.masked_dataset.positions.in_list == [[(2.0, 2.0), (3.0, 3.0)]]

        # Auto positioning is Off, but there are previous results with updated positions relative to the input
        # positions, so use those with their positions threshold.

        phase_imaging_7x7.meta_dataset.positions_threshold = 0.1
        phase_imaging_7x7.meta_dataset.auto_positions_factor = None

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=al.Coordinates(coordinates=[[(2.0, 2.0)]]),
            results=mock_pipeline.MockResults(
                positions=al.Coordinates(coordinates=[[(3.0, 3.0), (4.0, 4.0)]]),
                updated_positions_threshold=0.3,
            ),
        )

        assert analysis.masked_dataset.positions.in_list == [[(3.0, 3.0), (4.0, 4.0)]]

        # Test function is called for phase_inteferometer

        phase_interferometer_7.meta_dataset.positions_threshold = None
        phase_interferometer_7.meta_dataset.auto_positions_factor = 2.0

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=mask_7x7,
            positions=al.Coordinates(coordinates=[[(1.0, 1.0)]]),
            results=mock_pipeline.MockResults(
                updated_positions=al.Coordinates(coordinates=[[(1.0, 1.0)]]),
                updated_positions_threshold=0.3,
            ),
        )

        assert analysis.masked_dataset.positions.in_list == [[(1.0, 1.0)]]

    def test__auto_positions_update_threshold__uses_auto_update_factor(
        self,
        phase_imaging_7x7,
        phase_interferometer_7,
        imaging_7x7,
        interferometer_7,
        mask_7x7,
    ):

        # Auto positioning is OFF, so use input positions + threshold.

        phase_imaging_7x7.meta_dataset.positions_threshold = 0.1
        phase_imaging_7x7.meta_dataset.auto_positions_factor = None

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=al.Coordinates(coordinates=[[(1.0, 1.0)]]),
            results=mock_pipeline.MockResults(),
        )

        assert analysis.masked_dataset.positions_threshold == 0.1

        # Auto positioning is ON, but there are no previous results, so use input positions.

        phase_imaging_7x7.meta_dataset.auto_positions_factor = 1.0

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=al.Coordinates(coordinates=[[(1.0, 0.0), (-1.0, 0.0)]]),
            results=mock_pipeline.MockResults(),
        )

        assert analysis.masked_dataset.positions_threshold == 2.0

        phase_imaging_7x7.meta_dataset.positions_threshold = 0.2
        phase_imaging_7x7.meta_dataset.auto_positions_factor = 3.0

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7,
            positions=al.Coordinates(coordinates=[[(1.0, 0.0), (-1.0, 0.0)]]),
            results=mock_pipeline.MockResults(updated_positions_threshold=0.2),
        )

        assert analysis.masked_dataset.positions_threshold == 6.0

        # Auto positioning is ON, but positionos are None so no update.

        phase_imaging_7x7.meta_dataset.auto_positions_factor = 1.0
        phase_imaging_7x7.positions_threshold = None

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert analysis.masked_dataset.positions_threshold == None

    def test__pixelization_property_extracts_pixelization(self, imaging_7x7, mask_7x7):
        source_galaxy = al.Galaxy(redshift=0.5)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[source_galaxy], cosmology=cosmo.FLRW, phase_name="test_phase"
        )

        assert phase_imaging_7x7.meta_dataset.pixelization is None
        assert phase_imaging_7x7.meta_dataset.has_pixelization is False
        assert phase_imaging_7x7.meta_dataset.pixelizaition_is_model == False

        source_galaxy = al.Galaxy(
            redshift=0.5,
            pixelization=al.pix.Rectangular(),
            regularization=al.reg.Constant(),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[source_galaxy], cosmology=cosmo.FLRW, phase_name="test_phase"
        )

        assert isinstance(
            phase_imaging_7x7.meta_dataset.pixelization, al.pix.Rectangular
        )
        assert phase_imaging_7x7.meta_dataset.has_pixelization is True
        assert phase_imaging_7x7.meta_dataset.pixelizaition_is_model == False

        source_galaxy = al.GalaxyModel(
            redshift=0.5,
            pixelization=al.pix.Rectangular,
            regularization=al.reg.Constant,
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[source_galaxy], cosmology=cosmo.FLRW, phase_name="test_phase"
        )

        assert type(phase_imaging_7x7.meta_dataset.pixelization) == type(
            al.pix.Rectangular
        )
        assert phase_imaging_7x7.meta_dataset.has_pixelization is True
        assert phase_imaging_7x7.meta_dataset.pixelizaition_is_model == True

    def test__check_if_phase_uses_cluster_inversion(self):
        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is False

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular,
                    regularization=al.reg.Constant,
                ),
                source=al.GalaxyModel(redshift=1.0),
            ),
        )
        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is False

        source = al.GalaxyModel(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage,
            regularization=al.reg.Constant,
        )

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(lens=al.GalaxyModel(redshift=0.5), source=source),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is True

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is False

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular,
                    regularization=al.reg.Constant,
                ),
                source=al.GalaxyModel(redshift=1.0),
            ),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is False

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5),
                source=al.GalaxyModel(
                    redshift=1.0,
                    pixelization=al.pix.VoronoiBrightnessImage,
                    regularization=al.reg.Constant,
                ),
            ),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is True

    def test__use_border__determines_if_border_pixel_relocation_is_used(
        self, imaging_7x7, mask_7x7
    ):
        # noinspection PyTypeChecker

        lens_galaxy = al.Galaxy(
            redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=100.0)
        )
        source_galaxy = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.Rectangular(shape=(3, 3)),
            regularization=al.reg.Constant(coefficient=1.0),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            cosmology=cosmo.Planck15,
            phase_name="test_phase",
            inversion_uses_border=True,
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        analysis.masked_dataset.grid[4] = np.array([[500.0, 0.0]])

        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.masked_imaging_fit_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.grid[4][0] == pytest.approx(97.19584, 1.0e-2)
        assert fit.inversion.mapper.grid[4][1] == pytest.approx(-3.699999, 1.0e-2)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            cosmology=cosmo.Planck15,
            phase_name="test_phase",
            inversion_uses_border=False,
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        analysis.masked_dataset.grid[4] = np.array([300.0, 0.0])

        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.masked_imaging_fit_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.grid[4][0] == pytest.approx(200.0, 1.0e-4)

    def test__inversion_pixel_limit_computed_via_config_or_input(self,):
        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_imaging_7x7", inversion_pixel_limit=None
        )

        assert phase_imaging_7x7.meta_dataset.inversion_pixel_limit == 3000

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_imaging_7x7", inversion_pixel_limit=10
        )

        assert phase_imaging_7x7.meta_dataset.inversion_pixel_limit == 10

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_imaging_7x7", inversion_pixel_limit=2000
        )

        assert phase_imaging_7x7.meta_dataset.inversion_pixel_limit == 2000
