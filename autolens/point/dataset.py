from typing import List, Tuple, Optional, Union

import autoarray as aa


class PointDataset:
    def __init__(
        self,
        name: str,
        positions: Union[aa.Grid2DIrregular, List[List], List[Tuple]],
        positions_noise_map: Union[float, aa.ArrayIrregular, List[float]],
        fluxes: Optional[Union[aa.ArrayIrregular, List[float]]] = None,
        fluxes_noise_map: Optional[Union[float, aa.ArrayIrregular, List[float]]] = None,
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
            positions = aa.Grid2DIrregular(values=positions)

        self.positions = positions

        if isinstance(positions_noise_map, float):
            positions_noise_map = aa.ArrayIrregular(values=len(positions) * [positions_noise_map])

        if not isinstance(positions_noise_map, aa.ArrayIrregular):
            positions_noise_map = aa.ArrayIrregular(values=positions_noise_map)

        self.positions_noise_map = positions_noise_map

        if fluxes is not None:
            if not isinstance(fluxes, aa.ArrayIrregular):
                fluxes = aa.ArrayIrregular(values=fluxes)

        self.fluxes = fluxes

        if isinstance(fluxes_noise_map, float):
            fluxes_noise_map = aa.ArrayIrregular(values=len(fluxes) * [fluxes_noise_map])

        if fluxes_noise_map is not None:
            if not isinstance(fluxes_noise_map, aa.ArrayIrregular):
                fluxes_noise_map = aa.ArrayIrregular(values=fluxes_noise_map)

        self.fluxes_noise_map = fluxes_noise_map

    @property
    def info(self) -> str:
        """
        A dictionary representation of this instance.

        Arrays are represented as lists or lists of lists.
        """
        info = f"name : {self.name}\n"
        info += f"positions : {self.positions}\n"
        info += f"positions_noise_map : {self.positions_noise_map}\n"
        info += f"fluxes : {self.fluxes}\n"
        info += f"fluxes_noise_map : {self.fluxes_noise_map}\n"
        return info
