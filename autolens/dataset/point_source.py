from typing import List, Dict, Optional

from autoarray.structures.arrays import values
from autoarray.structures.grids.two_d import grid_2d_irregular

import json
import os
from os import path
import numpy as np


class PointSourceDataset:
    def __init__(
        self,
        name: str,
        positions: grid_2d_irregular.Grid2DIrregular,
        positions_noise_map: values.ValuesIrregular,
        fluxes: Optional[values.ValuesIrregular] = None,
        fluxes_noise_map: Optional[values.ValuesIrregular] = None,
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
        name
            The name of the point source dataset which is paired to a `PointSource` in the `Model`.
        positions
            The image-plane (y,x) positions of the point-source.
        positions_noise_map
            The noise-value of every (y,x) position, which is typically the pixel-scale of the data.
        fluxes
            The image-plane flux of each observed point-source of light.
        fluxes_noise_map
            The noise-value of every observed flux.
        """

        self.name = name
        self.positions = positions
        self.positions_noise_map = positions_noise_map
        self.fluxes = fluxes
        self.fluxes_noise_map = fluxes_noise_map

    @property
    def dict(self) -> dict:
        """
        A dictionary representation of this instance.

        Arrays are represented as lists or lists of lists.
        """
        return {
            "name": self.name,
            "positions": list(map(list, np.round(self.positions, 4))),
            "positions_noise_map": list(self.positions_noise_map),
            "fluxes": list(np.round(self.fluxes, 4))
            if self.fluxes is not None
            else None,
            "fluxes_noise_map": list(self.fluxes_noise_map)
            if self.fluxes_noise_map is not None
            else None,
        }

    @classmethod
    def from_dict(cls, dict_: dict) -> "PointSourceDataset":
        """
        Create a point source dataset from a dictionary representation.

        Parameters
        ----------
        dict_
            A dictionary. Arrays are represented as lists or lists of lists.

        Returns
        -------
        An instance
        """
        return cls(
            name=dict_["name"],
            positions=grid_2d_irregular.Grid2DIrregular(dict_["positions"]),
            positions_noise_map=values.ValuesIrregular(dict_["positions_noise_map"]),
            fluxes=values.ValuesIrregular(dict_["fluxes"])
            if dict_["fluxes"] is not None
            else None,
            fluxes_noise_map=values.ValuesIrregular(dict_["fluxes_noise_map"])
            if dict_["fluxes_noise_map"] is not None
            else None,
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
    def positions_list(self):
        return [
            point_source_dataset.positions
            for keys, point_source_dataset in self.items()
        ]

    @property
    def dicts(self) -> List[dict]:
        """
        A list of dictionaries representing this collection of point source
        datasets.
        """
        return [dataset.dict for dataset in self.values()]

    @classmethod
    def from_dicts(cls, dicts: List[dict]) -> List[PointSourceDataset]:
        """
        Create an instance from a list of dictionaries.

        Parameters
        ----------
        dicts
            Dictionaries, each representing one point source dataset.

        Returns
        -------
        A collection of point source datasets.
        """
        return cls(list(map(PointSourceDataset.from_dict, dicts)))

    @classmethod
    def from_json(cls, file_path):

        with open(file_path) as infile:
            dicts = json.load(infile)

        return cls.from_dicts(dicts=dicts)

    def output_to_json(self, file_path, overwrite=False):

        file_dir = os.path.split(file_path)[0]

        if not path.exists(file_dir):
            os.makedirs(file_dir)

        if overwrite and path.exists(file_path):
            os.remove(file_path)
        elif not overwrite and path.exists(file_path):
            raise FileExistsError(
                "The file ",
                file_path,
                " already exists. Set overwrite=True to overwrite this" "file",
            )

        with open(file_path, "w+") as f:
            json.dump(self.dicts, f, indent=4)
