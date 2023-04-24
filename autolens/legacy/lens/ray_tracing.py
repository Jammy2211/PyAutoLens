from typing import Dict, List, Optional

import autoarray as aa
import autogalaxy as ag

from autolens.lens.ray_tracing import Tracer as TracerBase


class Tracer(TracerBase):
    @classmethod
    def from_galaxies(
        cls,
        galaxies,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
        profiling_dict: Optional[Dict] = None,
    ):
        planes = ag.util.plane.planes_via_galaxies_from(
            galaxies=galaxies, profiling_dict=profiling_dict, plane_cls=ag.legacy.Plane
        )

        return cls(planes=planes, cosmology=cosmology, profiling_dict=profiling_dict)

    def hyper_noise_map_from(self, noise_map: aa.Array2D) -> aa.Array2D:
        return sum(self.hyper_noise_map_list_from(noise_map=noise_map))

    def hyper_noise_map_list_from(self, noise_map: aa.Array2D) -> List[aa.Array2D]:
        return [
            plane.hyper_noise_map_from(noise_map=noise_map) for plane in self.planes
        ]

    @property
    def contribution_map(self) -> Optional[aa.Array2D]:

        contribution_map_list = [i for i in self.contribution_map_list if i is not None]

        if contribution_map_list:
            return sum(contribution_map_list)

    @property
    def contribution_map_list(self) -> List[aa.Array2D]:
        contribution_map_list = []

        for plane in self.planes:
            if plane.contribution_map is not None:
                contribution_map_list.append(plane.contribution_map)

            else:
                contribution_map_list.append(None)

        return contribution_map_list
