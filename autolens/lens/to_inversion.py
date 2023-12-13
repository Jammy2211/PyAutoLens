from typing import Dict, List, Optional, Tuple, Type, Union

from autoconf import cached_property

import autoarray as aa
import autogalaxy as ag

from autoarray.inversion.inversion.factory import inversion_unpacked_from

from autolens.analysis.preloads import Preloads


class TracerToInversion(ag.AbstractToInversion):
    def __init__(
        self,
        tracer,
        dataset: Optional[Union[aa.Imaging, aa.Interferometer]] = None,
        data: Optional[Union[aa.Array2D, aa.Visibilities]] = None,
        noise_map: Optional[Union[aa.Array2D, aa.VisibilitiesNoiseMap]] = None,
        w_tilde: Optional[Union[aa.WTildeImaging, aa.WTildeInterferometer]] = None,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads=Preloads(),
        run_time_dict: Optional[Dict] = None,
    ):
        self.tracer = tracer

        super().__init__(
            dataset=dataset,
            data=data,
            noise_map=noise_map,
            w_tilde=w_tilde,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
            run_time_dict=run_time_dict,
        )

    @property
    def planes(self):
        return self.tracer.planes

    @cached_property
    @aa.profile_func
    def traced_grid_2d_list_of_inversion(self) -> List[aa.type.Grid2DLike]:
        return self.tracer.traced_grid_2d_list_from(grid=self.dataset.grid_pixelization)

    @cached_property
    def lp_linear_func_list_galaxy_dict(
        self,
    ) -> Dict[ag.LightProfileLinearObjFuncList, ag.Galaxy]:
        if not self.tracer.perform_inversion:
            return {}

        lp_linear_galaxy_dict_list = {}

        traced_grids_of_planes_list = self.tracer.traced_grid_2d_list_from(
            grid=self.dataset.grid
        )

        if self.dataset.blurring_grid is not None:
            traced_blurring_grids_of_planes_list = self.tracer.traced_grid_2d_list_from(
                grid=self.dataset.blurring_grid
            )
        else:
            traced_blurring_grids_of_planes_list = [None] * len(
                traced_grids_of_planes_list
            )

        for plane_index, plane in enumerate(self.planes):
            plane_to_inversion = ag.PlaneToInversion(
                plane=plane,
                dataset=self.dataset,
                grid=traced_grids_of_planes_list[plane_index],
                blurring_grid=traced_blurring_grids_of_planes_list[plane_index],
                noise_map=self.noise_map,
            )

            lp_linear_galaxy_dict_of_plane = (
                plane_to_inversion.lp_linear_func_list_galaxy_dict
            )

            lp_linear_galaxy_dict_list = {
                **lp_linear_galaxy_dict_list,
                **lp_linear_galaxy_dict_of_plane,
            }

        return lp_linear_galaxy_dict_list

    def cls_pg_list_from(self, cls: Type) -> List:
        return [plane.cls_list_from(cls=cls) for plane in self.planes]

    @cached_property
    def adapt_galaxy_image_pg_list(self) -> List:
        return [
            plane.adapt_galaxies_with_pixelization_image_list for plane in self.planes
        ]

    @cached_property
    @aa.profile_func
    def image_plane_mesh_grid_pg_list(self) -> List[List]:
        """
        Specific pixelizations, like the `VoronoiMagnification`, begin by determining what will become its the
        source-pixel centres by calculating them  in the image-plane. The `VoronoiBrightnessImage` pixelization
        performs a KMeans clustering.
        """

        image_plane_mesh_grid_list_of_planes = []

        for plane in self.planes:
            plane_to_inversion = ag.PlaneToInversion(
                plane=plane,
                grid_pixelization=self.dataset.grid,
                settings_pixelization=self.settings_pixelization,
                noise_map=self.noise_map,
            )

            image_plane_mesh_grid_list = (
                plane_to_inversion.image_plane_mesh_grid_list
            )
            image_plane_mesh_grid_list_of_planes.append(image_plane_mesh_grid_list)

        return image_plane_mesh_grid_list_of_planes

    @cached_property
    @aa.profile_func
    def traced_mesh_grid_pg_list(self) -> Tuple[List[List], List[List]]:
        """
        Ray-trace the sparse image plane grid used to define the source-pixel centres by calculating the deflection
        angles at (y,x) coordinate on the grid from the galaxy mass profiles and then ray-trace them from the
        image-plane to the source plane.
        """
        if (
            self.preloads.image_plane_mesh_grid_pg_list is None
            or self.settings_pixelization.is_stochastic
        ):
            image_plane_mesh_grid_pg_list = self.image_plane_mesh_grid_pg_list

        else:
            image_plane_mesh_grid_pg_list = (
                self.preloads.image_plane_mesh_grid_pg_list
            )

        traced_mesh_grid_pg_list = []

        for plane_index, plane in enumerate(self.planes):
            if image_plane_mesh_grid_pg_list[plane_index] is None:
                traced_mesh_grid_pg_list.append(None)
            else:
                traced_mesh_grids_list = []

                for image_plane_mesh_grid in image_plane_mesh_grid_pg_list[
                    plane_index
                ]:
                    try:
                        traced_mesh_grids_list.append(
                            self.tracer.traced_grid_2d_list_from(
                                grid=image_plane_mesh_grid
                            )[plane_index]
                        )
                    except AttributeError:
                        traced_mesh_grids_list.append(None)

                traced_mesh_grid_pg_list.append(traced_mesh_grids_list)

        return traced_mesh_grid_pg_list, image_plane_mesh_grid_pg_list

    @cached_property
    def mapper_galaxy_dict(self) -> Dict[aa.AbstractMapper, ag.Galaxy]:
        mapper_galaxy_dict = {}

        if self.preloads.traced_grids_of_planes_for_inversion is None:
            traced_grids_of_planes_list = self.traced_grid_2d_list_of_inversion
        else:
            traced_grids_of_planes_list = (
                self.preloads.traced_grids_of_planes_for_inversion
            )

        if self.preloads.traced_mesh_grids_list_of_planes is None:
            (
                traced_mesh_grids_list_of_planes,
                image_plane_mesh_grid_list,
            ) = self.traced_mesh_grid_pg_list
        else:
            traced_mesh_grids_list_of_planes = (
                self.preloads.traced_mesh_grids_list_of_planes
            )
            image_plane_mesh_grid_list = self.preloads.image_plane_mesh_grid_list

        for plane_index, plane in enumerate(self.planes):
            if plane.has(cls=aa.Pixelization):
                plane_to_inversion = ag.PlaneToInversion(
                    plane=plane,
                    grid_pixelization=traced_grids_of_planes_list[plane_index],
                    settings_pixelization=self.settings_pixelization,
                    preloads=self.preloads,
                    noise_map=self.noise_map,
                )

                galaxies_with_pixelization_list = plane.galaxies_with_cls_list_from(
                    cls=aa.Pixelization
                )

                for mapper_index in range(
                    len(traced_mesh_grids_list_of_planes[plane_index])
                ):
                    pixelization_list = self.cls_pg_list_from(cls=aa.Pixelization)

                    mapper = plane_to_inversion.mapper_from(
                        mesh=pixelization_list[plane_index][mapper_index].mesh,
                        regularization=pixelization_list[plane_index][
                            mapper_index
                        ].regularization,
                        source_plane_mesh_grid=traced_mesh_grids_list_of_planes[
                            plane_index
                        ][mapper_index],
                        image_plane_mesh_grid=image_plane_mesh_grid_list[plane_index][
                            mapper_index
                        ],
                        adapt_galaxy_image=self.adapt_galaxy_image_pg_list[plane_index][
                            mapper_index
                        ],
                    )

                    galaxy = galaxies_with_pixelization_list[mapper_index]

                    mapper_galaxy_dict[mapper] = galaxy

        return mapper_galaxy_dict

    @cached_property
    def inversion(self):
        inversion = inversion_unpacked_from(
            dataset=self.dataset,
            data=self.data,
            noise_map=self.noise_map,
            w_tilde=self.w_tilde,
            linear_obj_list=self.linear_obj_list,
            settings=self.settings_inversion,
            preloads=self.preloads,
            run_time_dict=self.tracer.run_time_dict,
        )

        inversion.linear_obj_galaxy_dict = self.linear_obj_galaxy_dict

        return inversion
