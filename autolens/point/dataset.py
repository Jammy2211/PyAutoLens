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
        time_delays: Optional[Union[aa.ArrayIrregular, List[float]]] = None,
        time_delays_noise_map: Optional[
            Union[float, aa.ArrayIrregular, List[float]]
        ] = None,
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
            The noise-value of every observed flux, which is typically measured from the pixel values of the pixel
            containing the point source after convolution with the PSF.
        time_delays
            The time delays of each observed point-source of light in days.
        time_delays_noise_map
            The noise-value of every observed time delay, which is typically measured from the time delay analysis.
        """

        self.name = name

        # Ensure positions is a Grid2DIrregular
        self.positions = (
            positions
            if isinstance(positions, aa.Grid2DIrregular)
            else aa.Grid2DIrregular(values=positions)
        )

        # Ensure positions_noise_map is an ArrayIrregular
        if isinstance(positions_noise_map, float):
            positions_noise_map = [positions_noise_map] * len(self.positions)

        self.positions_noise_map = (
            positions_noise_map
            if isinstance(positions_noise_map, aa.ArrayIrregular)
            else aa.ArrayIrregular(values=positions_noise_map)
        )

        def convert_to_array_irregular(values):
            """
            Convert data to ArrayIrregular if it is not already.
            """
            return (
                aa.ArrayIrregular(values=values)
                if values is not None and not isinstance(values, aa.ArrayIrregular)
                else values
            )

        # Convert fluxes, time delays and their noise maps to ArrayIrregular if provided as values and not already this type

        self.fluxes = convert_to_array_irregular(fluxes)
        self.fluxes_noise_map = convert_to_array_irregular(fluxes_noise_map)
        self.time_delays = convert_to_array_irregular(time_delays)
        self.time_delays_noise_map = convert_to_array_irregular(time_delays_noise_map)

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
        info += f"time_delays : {self.time_delays}\n"
        info += f"time_delays_noise_map : {self.time_delays_noise_map}\n"
        return info

    def extent_from(self, buffer: float = 0.1):
        y_max = max(self.positions[:, 0]) + buffer
        y_min = min(self.positions[:, 0]) - buffer
        x_max = max(self.positions[:, 1]) + buffer
        x_min = min(self.positions[:, 1]) - buffer

        return [y_min, y_max, x_min, x_max]
