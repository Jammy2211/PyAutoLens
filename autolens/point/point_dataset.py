import json
import numpy as np
import os
from os import path
from typing import List, Tuple, Dict, Optional, Union

import autoarray as aa


class PointDataset:
    def __init__(
        self,
        name: str,
        positions: Union[aa.Grid2DIrregular, List[List], List[Tuple]],
        positions_noise_map: Union[aa.ValuesIrregular, List[float]],
        fluxes: Optional[Union[aa.ValuesIrregular, List[float]]] = None,
        fluxes_noise_map: Optional[Union[aa.ValuesIrregular, List[float]]] = None,
    ):
        """
        A collection of the data component that can be used for point-source model-fitting, for example fitting the
        observed positions of a a strongly lensed quasar or supernovae or in strong lens cluster modeling, where
        there may be many tens or hundreds of individual source galaxies each of which are modeled as a point source.

        The name of the dataset is required for point-source model-fitting, as it pairs a point-source dataset with
        its corresponding point-source in the model-fit. For example, if a dataset has the name `source_1`, it will
        be paired with the `Point` model-component which has the name `source_1`. If a dataset component is not
        successfully paired with a model-component, an error is raised.

        Parameters
        ----------
        name
            The name of the point source dataset which is paired to a `Point` in the `Model`.
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

        if not isinstance(positions, aa.Grid2DIrregular):
            positions = aa.Grid2DIrregular(grid=positions)

        self.positions = positions

        if not isinstance(positions_noise_map, aa.ValuesIrregular):
            positions_noise_map = aa.ValuesIrregular(values=positions_noise_map)

        self.positions_noise_map = positions_noise_map

        if fluxes is not None:
            if not isinstance(fluxes, aa.ValuesIrregular):
                fluxes = aa.ValuesIrregular(values=fluxes)

        self.fluxes = fluxes

        if fluxes_noise_map is not None:
            if not isinstance(fluxes_noise_map, aa.ValuesIrregular):
                fluxes_noise_map = aa.ValuesIrregular(values=fluxes_noise_map)

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
    def from_dict(cls, dict_: dict) -> "PointDataset":
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
            positions=aa.Grid2DIrregular(dict_["positions"]),
            positions_noise_map=aa.ValuesIrregular(dict_["positions_noise_map"]),
            fluxes=aa.ValuesIrregular(dict_["fluxes"])
            if dict_["fluxes"] is not None
            else None,
            fluxes_noise_map=aa.ValuesIrregular(dict_["fluxes_noise_map"])
            if dict_["fluxes_noise_map"] is not None
            else None,
        )


class PointDict(dict):
    def __init__(self, point_dataset_list: List[PointDataset]):
        """
        A dictionary containing the entire point-source dataset, which could be many instances of
        the `PointDataset` object.

        This dictionary uses the `name` of the `PointDataset` to act as the key of every entry of the dictionary,
        making it straight forward to access the attributes based on the dataset name.

        Parameters
        ----------
        point_dataset_list
            A list of all point-source datasets that are to be added to the point-source dictionary.

        Returns
        -------
        Dict[PointDataset]
            A dictionary where the keys are the `name` entries of each `PointDataset` and the values are
            the corresponding instance of the `PointDataset` class.
        """

        super().__init__()

        for point_dataset in point_dataset_list:

            self[point_dataset.name] = point_dataset

    @property
    def positions_list(self):
        return [point_dataset.positions for keys, point_dataset in self.items()]

    @property
    def dicts(self) -> List[dict]:
        """
        A list of dictionaries representing this collection of point source
        datasets.
        """
        return [dataset.dict for dataset in self.values()]

    @classmethod
    def from_dicts(cls, dicts: List[dict]) -> List[PointDataset]:
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
        return cls(map(PointDataset.from_dict, dicts))

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
