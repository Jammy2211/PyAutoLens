from autolens import decorator_util

import numpy as np


class Transformer(object):
    def __init__(self, uv_wavelengths, grid_radians, preload_transform=True):

        self.uv_wavelengths = uv_wavelengths
        self.grid_radians = grid_radians

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = grid_radians.shape[0]

        self.preload_transform = preload_transform

        if preload_transform:

            self.preload_real_transforms = self.preload_real_transforms(
                grid_radians=grid_radians,
                uv_wavelengths=uv_wavelengths,
                total_image_pixels=self.total_image_pixels,
            )

            self.preload_imaginary_transforms = self.preload_imaginary_transforms(
                grid_radians=grid_radians,
                uv_wavelengths=uv_wavelengths,
                total_image_pixels=self.total_image_pixels,
            )

    def real_visibilities_from_image_1d(self, image_1d):

        if self.preload_transform:

            return self.real_visibilities_via_preload_jit(
                image_1d=image_1d,
                preloaded_reals=self.preload_real_transforms,
                total_visibilities=self.total_visibilities,
                total_image_pixels=self.total_image_pixels,
            )

        else:

            return self.real_visibilities_jit(
                image_1d=image_1d,
                grid_radians=self.grid_radians,
                uv_wavelengths=self.uv_wavelengths,
                total_visibilities=self.total_visibilities,
                total_image_pixels=self.total_image_pixels,
            )

    @staticmethod
    @decorator_util.jit()
    def preload_real_transforms(grid_radians, uv_wavelengths, total_image_pixels):

        preloaded_real_transforms = np.zeros(
            shape=(total_image_pixels, uv_wavelengths.shape[0])
        )

        for i in range(total_image_pixels):
            for j in range(uv_wavelengths.shape[0]):
                preloaded_real_transforms[i, j] += np.cos(
                    -2.0
                    * np.pi
                    * (
                        grid_radians[i, 1] * uv_wavelengths[j, 0]
                        - grid_radians[i, 0] * uv_wavelengths[j, 1]
                    )
                )

        return preloaded_real_transforms

    @staticmethod
    @decorator_util.jit()
    def real_visibilities_via_preload_jit(
        image_1d, preloaded_reals, total_visibilities, total_image_pixels
    ):

        real_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                real_visibilities[j] += image_1d[i] * preloaded_reals[i, j]

        return real_visibilities

    @staticmethod
    @decorator_util.jit()
    def real_visibilities_jit(
        image_1d, grid_radians, uv_wavelengths, total_visibilities, total_image_pixels
    ):

        real_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                real_visibilities[j] += image_1d[i] * np.cos(
                    -2.0
                    * np.pi
                    * (
                        grid_radians[i, 1] * uv_wavelengths[j, 0]
                        - grid_radians[i, 0] * uv_wavelengths[j, 1]
                    )
                )

        return real_visibilities

    def imaginary_visibilities_from_image_1d(self, image_1d):

        if self.preload_transform:

            return self.imaginary_visibilities_via_preload_jit(
                image_1d=image_1d,
                preloaded_imaginarys=self.preload_imaginary_transforms,
                total_visibilities=self.total_visibilities,
                total_image_pixels=self.total_image_pixels,
            )

        else:

            return self.imaginary_visibilities_jit(
                image_1d=image_1d,
                grid_radians=self.grid_radians,
                uv_wavelengths=self.uv_wavelengths,
                total_visibilities=self.total_visibilities,
                total_image_pixels=self.total_image_pixels,
            )

    @staticmethod
    @decorator_util.jit()
    def preload_imaginary_transforms(grid_radians, uv_wavelengths, total_image_pixels):

        preloaded_imaginary_transforms = np.zeros(
            shape=(total_image_pixels, uv_wavelengths.shape[0])
        )

        for i in range(total_image_pixels):
            for j in range(uv_wavelengths.shape[0]):
                preloaded_imaginary_transforms[i, j] += np.sin(
                    -2.0
                    * np.pi
                    * (
                        grid_radians[i, 1] * uv_wavelengths[j, 0]
                        - grid_radians[i, 0] * uv_wavelengths[j, 1]
                    )
                )

        return preloaded_imaginary_transforms

    @staticmethod
    @decorator_util.jit()
    def imaginary_visibilities_via_preload_jit(
        image_1d, preloaded_imaginarys, total_visibilities, total_image_pixels
    ):

        imaginary_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                imaginary_visibilities[j] += image_1d[i] * preloaded_imaginarys[i, j]

        return imaginary_visibilities

    @staticmethod
    @decorator_util.jit()
    def imaginary_visibilities_jit(
        image_1d, grid_radians, uv_wavelengths, total_visibilities, total_image_pixels
    ):

        imaginary_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                imaginary_visibilities[j] += image_1d[i] * np.sin(
                    -2.0
                    * np.pi
                    * (
                        grid_radians[i, 1] * uv_wavelengths[j, 0]
                        - grid_radians[i, 0] * uv_wavelengths[j, 1]
                    )
                )

        return imaginary_visibilities

    def visibilities_from_image_1d(self, image_1d):

        real_visibilities = self.real_visibilities_from_image_1d(image_1d=image_1d)
        imaginary_visibilities = self.imaginary_visibilities_from_image_1d(
            image_1d=image_1d
        )

        return np.stack((real_visibilities, imaginary_visibilities), axis=-1)
