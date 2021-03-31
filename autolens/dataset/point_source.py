from typing import List, Dict

from autoarray.structures.arrays import values
from autoarray.structures.grids.two_d import grid_2d_irregular


class PointSourceDataset:
    def __init__(
            self,
            name: str,
            positions: grid_2d_irregular.Grid2DIrregular = None,
            positions_noise_map: values.ValuesIrregular = None,
            fluxes: values.ValuesIrregular = None,
            fluxes_noise_map: values.ValuesIrregular = None,
    ):
        """
        A collection of the data component that can be used for point-source model-fitting, for example fitting the
        observed positions of a a strongly lensed quasar or supernovae or in strong lens cluster modeling, where
        there may be many tens or hundreds of individual source galaxies each of which are modeled as a point source.

        The name of the dataset is required for point-source model-fitting, as it pairs a point-source dataset with
        its corresponding point-source in the model-fit. For example, if a dataset has the name `source_1`, it will
        be paired with the `PointSource` model-component which has the name `source_1`. If a dataset component is not
        successfully paired with a model-component, an error is raised.

        Parameters
        ----------
        name : str
            The name of the point source dataset which is paired to a `PointSource` in the `Model`.
        positions : grid_2d_irregular.Grid2DIrregular
            The image-plane (y,x) positions of the point-source.
        positions_noise_map : values.ValuesIrregular
            The noise-value of every (y,x) position, which is typically the pixel-scale of the data.
        fluxes : values.ValuesIrregular
            The image-plane flux of each observed point-source of light.
        fluxes_noise_map : values.ValuesIrregular
            The noise-value of every observed flux.
        """

        self.name = name
        self.positions = positions
        self.positions_noise_map = positions_noise_map
        self.fluxes = fluxes
        self.fluxes_noise_map = fluxes_noise_map

    @property
    def dict(self):
        return {
            "name": self.name,
            "positions": list(map(list, self.positions)),
            "positions_noise_map": list(self.positions_noise_map),
            "fluxes": list(self.fluxes),
            "fluxes_noise_map": list(self.fluxes_noise_map),
        }

    @classmethod
    def from_dict(
            cls,
            dict_
    ):
        return cls(
            name=dict_["name"],
            positions=grid_2d_irregular.Grid2DIrregular(
                dict_["positions"]
            ),
            positions_noise_map=values.ValuesIrregular(
                dict_["positions_noise_map"]
            ),
            fluxes=values.ValuesIrregular(
                dict_["fluxes"]
            ),
            fluxes_noise_map=values.ValuesIrregular(
                dict_["fluxes_noise_map"]
            )
        )


class PointSourceDict(dict):
    def __init__(self, point_source_dataset_list: List[PointSourceDataset]):
        """
        A dictionary containing the entire point-source dataset, which could be many instances of
        the `PointSourceDataset` object.

        This dictionary uses the `name` of the `PointSourceDataset` to act as the key of every entry of the dictionary,
        making it straight forward to access the attributes based on the dataset name.

        Parameters
        ----------
        point_source_dataset_list : [PointSourceDataset]
            A list of all point-source datasets that are to be added to the point-source dictionary.

        Returns
        -------
        Dict[PointSourceDataset]
            A dictionary where the keys are the `name` entries of each `PointSourceDataset` and the values are
            the corresponding instance of the `PointSourceDataset` class.
        """

        super().__init__()

        for point_source_dataset in point_source_dataset_list:

            self[point_source_dataset.name] = point_source_dataset

    @property
    def dicts(self):
        return [
            dataset.dict
            for dataset
            in self.values()
        ]

    @classmethod
    def from_dicts(
            cls,
            dicts
    ):
        return cls(
            list(map(
                PointSourceDataset.from_dict,
                dicts
            ))
        )
