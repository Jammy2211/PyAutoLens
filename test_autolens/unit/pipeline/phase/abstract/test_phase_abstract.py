import pytest

import autolens as al
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)


class TestSetup:
    def test__uses_pixelization_preload_grids_if_possible(self, imaging_7x7, mask_7x7):
        phase_dataset_7x7 = al.PhaseImaging(phase_name="test_phase")

        analysis = phase_dataset_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert analysis.masked_dataset.preload_sparse_grids_of_planes is None
