from typing import Dict, List, Optional, Tuple, Type, Union

import autoarray as aa
import autogalaxy as ag

from autoarray.inversion.inversion.factory import inversion_imaging_unpacked_from
from autoarray.inversion.inversion.factory import inversion_interferometer_unpacked_from

from autolens.analysis.preloads import Preloads


class TracerToInversion:
    def __init__(self, tracer, profiling_dict: Optional[Dict] = None):

        self.tracer = tracer
        self.profiling_dict = profiling_dict

    @property
    def planes(self):
        return self.tracer.planes

    @aa.profile_func
    def traced_grid_2d_list_of_inversion_from(
        self, grid: aa.type.Grid2DLike
    ) -> List[aa.type.Grid2DLike]:
        return self.tracer.traced_grid_2d_list_from(grid=grid)

    def lp_linear_func_galaxy_dict_from(
        self,
        grid: aa.type.Grid2DLike,
        blurring_grid: aa.type.Grid1D2DLike,
        convolver: Optional[aa.Convolver] = None,
        preloads=Preloads(),
    ) -> Dict[ag.LightProfileLinearObjFunc, ag.Galaxy]:

        if not self.tracer.has(cls=ag.lp_linear.LightProfileLinear):
            return {}

        lp_linear_galaxy_dict_list = {}

        traced_grids_of_planes_list = self.tracer.traced_grid_2d_list_from(grid=grid)

        if blurring_grid is not None:
            traced_blurring_grids_of_planes_list = self.tracer.traced_grid_2d_list_from(
                grid=blurring_grid
            )
        else:
            traced_blurring_grids_of_planes_list = [None] * len(
                traced_grids_of_planes_list
            )

        for (plane_index, plane) in enumerate(self.planes):

            lp_linear_galaxy_dict_of_plane = plane.to_inversion.lp_linear_func_galaxy_dict_from(
                source_grid_slim=traced_grids_of_planes_list[plane_index],
                source_blurring_grid_slim=traced_blurring_grids_of_planes_list[
                    plane_index
                ],
                convolver=convolver,
            )

            lp_linear_galaxy_dict_list = {
                **lp_linear_galaxy_dict_list,
                **lp_linear_galaxy_dict_of_plane,
            }

        return lp_linear_galaxy_dict_list

    def cls_pg_list_from(self, cls: Type) -> List[List]:
        return [plane.cls_list_from(cls=cls) for plane in self.planes]

    @property
    def hyper_galaxy_image_pg_list(self) -> List[List]:
        return [
            plane.hyper_galaxies_with_pixelization_image_list for plane in self.planes
        ]

    @aa.profile_func
    def sparse_image_plane_grid_pg_list_from(
        self, grid: aa.type.Grid2DLike, settings_pixelization=aa.SettingsPixelization()
    ) -> List[List]:
        """
        Specific pixelizations, like the `VoronoiMagnification`, begin by determining what will become its the
        source-pixel centres by calculating them  in the image-plane. The `VoronoiBrightnessImage` pixelization
        performs a KMeans clustering.
        """

        sparse_image_plane_grid_list_of_planes = []

        for plane in self.planes:
            sparse_image_plane_grid_list = plane.to_inversion.sparse_image_plane_grid_list_from(
                grid=grid, settings_pixelization=settings_pixelization
            )
            sparse_image_plane_grid_list_of_planes.append(sparse_image_plane_grid_list)

        return sparse_image_plane_grid_list_of_planes

    @aa.profile_func
    def traced_sparse_grid_pg_list_from(
        self,
        grid: aa.type.Grid2DLike,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=Preloads(),
    ) -> Tuple[List[List], List[List]]:
        """
        Ray-trace the sparse image plane grid used to define the source-pixel centres by calculating the deflection
        angles at (y,x) coordinate on the grid from the galaxy mass profiles and then ray-trace them from the
        image-plane to the source plane.
        """
        if (
            preloads.sparse_image_plane_grid_pg_list is None
            or settings_pixelization.is_stochastic
        ):

            sparse_image_plane_grid_pg_list = self.sparse_image_plane_grid_pg_list_from(
                grid=grid, settings_pixelization=settings_pixelization
            )

        else:

            sparse_image_plane_grid_pg_list = preloads.sparse_image_plane_grid_pg_list

        traced_sparse_grid_pg_list = []

        for (plane_index, plane) in enumerate(self.planes):

            if sparse_image_plane_grid_pg_list[plane_index] is None:
                traced_sparse_grid_pg_list.append(None)
            else:

                traced_sparse_grids_list = []

                for sparse_image_plane_grid in sparse_image_plane_grid_pg_list[
                    plane_index
                ]:

                    try:
                        traced_sparse_grids_list.append(
                            self.tracer.traced_grid_2d_list_from(
                                grid=sparse_image_plane_grid
                            )[plane_index]
                        )
                    except AttributeError:
                        traced_sparse_grids_list.append(None)

                traced_sparse_grid_pg_list.append(traced_sparse_grids_list)

        return traced_sparse_grid_pg_list, sparse_image_plane_grid_pg_list

    def mapper_galaxy_dict_from(
        self,
        grid: aa.type.Grid2DLike,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=Preloads(),
    ) -> Dict[aa.AbstractMapper, ag.Galaxy]:

        mapper_galaxy_dict = {}

        if preloads.traced_grids_of_planes_for_inversion is None:
            traced_grids_of_planes_list = self.traced_grid_2d_list_of_inversion_from(
                grid=grid
            )
        else:
            traced_grids_of_planes_list = preloads.traced_grids_of_planes_for_inversion

        if preloads.traced_sparse_grids_list_of_planes is None:
            traced_sparse_grids_list_of_planes, sparse_image_plane_grid_list = self.traced_sparse_grid_pg_list_from(
                grid=grid,
                settings_pixelization=settings_pixelization,
                preloads=preloads,
            )
        else:
            traced_sparse_grids_list_of_planes = (
                preloads.traced_sparse_grids_list_of_planes
            )
            sparse_image_plane_grid_list = preloads.sparse_image_plane_grid_list

        for (plane_index, plane) in enumerate(self.planes):

            if plane.has(cls=aa.pix.Pixelization):

                galaxies_with_pixelization_list = plane.galaxies_with_cls_list_from(
                    cls=aa.pix.Pixelization
                )

                for mapper_index in range(
                    len(traced_sparse_grids_list_of_planes[plane_index])
                ):

                    mapper = plane.to_inversion.mapper_from(
                        source_grid_slim=traced_grids_of_planes_list[plane_index],
                        source_pixelization_grid=traced_sparse_grids_list_of_planes[
                            plane_index
                        ][mapper_index],
                        data_pixelization_grid=sparse_image_plane_grid_list[
                            plane_index
                        ][mapper_index],
                        pixelization=self.cls_pg_list_from(cls=aa.pix.Pixelization)[
                            plane_index
                        ][mapper_index],
                        hyper_galaxy_image=self.hyper_galaxy_image_pg_list[plane_index][
                            mapper_index
                        ],
                        settings_pixelization=settings_pixelization,
                        preloads=preloads,
                    )

                    galaxy = galaxies_with_pixelization_list[mapper_index]

                    mapper_galaxy_dict[mapper] = galaxy

        return mapper_galaxy_dict

    def linear_obj_galaxy_dict_from(
        self,
        dataset: Union[aa.Imaging, aa.Interferometer],
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        preloads: Preloads = Preloads(),
    ) -> Dict[Union[ag.LightProfileLinearObjFunc, aa.AbstractMapper], ag.Galaxy]:

        lp_linear_func_galaxy_dict = self.lp_linear_func_galaxy_dict_from(
            grid=dataset.grid,
            blurring_grid=dataset.blurring_grid,
            convolver=dataset.convolver,
        )

        if preloads.mapper_galaxy_dict is None:

            mapper_galaxy_dict = self.mapper_galaxy_dict_from(
                grid=dataset.grid_pixelized,
                settings_pixelization=settings_pixelization,
                preloads=preloads,
            )

        else:

            mapper_galaxy_dict = preloads.mapper_galaxy_dict

        return {**lp_linear_func_galaxy_dict, **mapper_galaxy_dict}

    def inversion_imaging_from(
        self,
        dataset: aa.Imaging,
        image: aa.Array2D,
        noise_map: aa.Array2D,
        w_tilde: aa.WTildeImaging,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads: Preloads = Preloads(),
    ):

        linear_obj_galaxy_dict = self.linear_obj_galaxy_dict_from(
            dataset=dataset,
            settings_pixelization=settings_pixelization,
            preloads=preloads,
        )

        linear_obj_list = list(linear_obj_galaxy_dict.keys())

        inversion = inversion_imaging_unpacked_from(
            image=image,
            noise_map=noise_map,
            convolver=dataset.convolver,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.tracer.cls_list_from(cls=aa.reg.Regularization),
            settings=settings_inversion,
            preloads=preloads,
            profiling_dict=self.tracer.profiling_dict,
        )

        inversion.linear_obj_galaxy_dict = linear_obj_galaxy_dict

        return inversion

    def inversion_interferometer_from(
        self,
        dataset: aa.Interferometer,
        visibilities: aa.Visibilities,
        noise_map: aa.VisibilitiesNoiseMap,
        w_tilde,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads: Preloads = Preloads(),
    ):

        linear_obj_galaxy_dict = self.linear_obj_galaxy_dict_from(
            dataset=dataset,
            settings_pixelization=settings_pixelization,
            preloads=preloads,
        )

        linear_obj_list = list(linear_obj_galaxy_dict.keys())

        print(self.tracer.cls_list_from(cls=aa.reg.Regularization))

        inversion = inversion_interferometer_unpacked_from(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=dataset.transformer,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.tracer.cls_list_from(cls=aa.reg.Regularization),
            settings=settings_inversion,
            profiling_dict=self.tracer.profiling_dict,
        )

        inversion.linear_obj_galaxy_dict = linear_obj_galaxy_dict

        return inversion
