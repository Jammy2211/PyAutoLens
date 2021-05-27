import autolens as al
import pytest


@pytest.fixture(name="dataset_dict")
def make_dataset_dict():
    return {
        "name": "name",
        "positions": [[1, 2]],
        "positions_noise_map": [1],
        "fluxes": [2],
        "fluxes_noise_map": [3],
    }


@pytest.fixture(name="dataset")
def make_dataset():
    return al.PointDataset(
        "name",
        positions=al.Grid2DIrregular([(1, 2)]),
        positions_noise_map=al.ValuesIrregular([1]),
        fluxes=al.ValuesIrregular([2]),
        fluxes_noise_map=al.ValuesIrregular([3]),
    )


class TestDataset:
    def test_to_dict(self, dataset_dict, dataset):
        assert dataset.dict == dataset_dict

    def test_from_dict(self, dataset_dict, dataset):
        dataset_ = al.PointDataset.from_dict(dataset_dict)
        assert (dataset_.positions == dataset.positions).all()
        assert (dataset_.positions_noise_map == dataset.positions_noise_map).all()
        assert (dataset_.fluxes == dataset.fluxes).all()
        assert (dataset_.fluxes_noise_map == dataset.fluxes_noise_map).all()


class TestDict:
    def test_dicts(self, dataset, dataset_dict):
        point_dict = al.PointDict([dataset])
        assert point_dict.dicts == [dataset_dict]

    def test_from_dicts(self, dataset, dataset_dict):
        point_dict = al.PointDict.from_dicts([dataset_dict])
        assert len(point_dict) == 1
        assert dataset.name in point_dict
