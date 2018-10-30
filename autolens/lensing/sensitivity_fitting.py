import numpy as np

from autolens import exc
from autolens.imaging import image as im
from autolens.inversion import inversions
from autolens.fitting import fitting
from autolens.fitting import fitting_data
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing

class AbstractSensitivityFit(object):

    def __init__(self, tracer_normal, tracer_sensitive):

        self.tracer_normal = tracer_normal
        self.tracer_sensitive = tracer_sensitive


class SensitivityProfileFit(AbstractSensitivityFit):

    def __init__(self, lensing_images, tracer_normal, tracer_sensitive, add_noise=True, noise_seed=-1):
        """
        Class to evaluate the fit between a model described by a tracer_normal and an actual lensing_image.

        Parameters
        ----------
        lensing_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer_normal: ray_tracing.AbstractTracer
            An object describing the model
        """
        AbstractSensitivityFit.__init__(self=self, tracer_normal=tracer_normal,
                                        tracer_sensitive=tracer_sensitive)

        self.sensitivity_images = sensitivity_images_from_lensing_images_and_tracer_sensitive(
            lensing_images=lensing_images, tracer_sensitive=tracer_sensitive, add_noise=add_noise,
            noise_seed=noise_seed)
        
        self.fit_normal = fitting.AbstractProfileFit(fitting_images=self.sensitivity_images,
                                                     _images=tracer_normal._image_plane_images,
                                                     _blurring_images=tracer_normal._image_plane_blurring_images)
        
        self.fit_sensitive = fitting.AbstractProfileFit(fitting_images=self.sensitivity_images,
                                                        _images=tracer_sensitive._image_plane_images,
                                                        _blurring_images=tracer_sensitive._image_plane_blurring_images)

    @classmethod
    def fast_likelihood(cls, lensing_images, tracer_normal, tracer_sensitive, add_noise=True, noise_seed=-1):
        """
        Fast calculation of likelihood

        Parameters
        ----------
        lensing_images: [li.LensingImage]
            An lensing_image that has been masked for efficiency
        tracer_normal: ray_tracing.AbstractTracer
            An object describing the model
        """

        sensitivity_images = sensitivity_images_from_lensing_images_and_tracer_sensitive(lensing_images=lensing_images,
                                                                                         tracer_sensitive=tracer_sensitive, add_noise=add_noise, noise_seed=noise_seed)

        convolvers = list(map(lambda lensing_image : lensing_image.convolver_image, lensing_images))
        _noise_maps = list(map(lambda lensing_image : lensing_image.noise_map, lensing_images))
        
        _model_images_normal = fitting.blur_images_including_blurring_regions(images=tracer_normal._image_plane_images,
                                     blurring_images=tracer_normal._image_plane_blurring_images, convolvers=convolvers)
        
        _residuals_normal = fitting.residuals_from_datas_and_model_datas(datas=sensitivity_images, 
                                                                         model_datas=_model_images_normal)
        
        _chi_squareds_normal = fitting.chi_squareds_from_residuals_and_noise_maps(residuals=_residuals_normal, 
                                                                                  noise_maps=_noise_maps)
        
        chi_squared_terms_normal = fitting.chi_squared_terms_from_chi_squareds(chi_squareds=_chi_squareds_normal)
        
        noise_terms_normal = fitting.noise_terms_from_noise_maps(noise_maps=_noise_maps)
        likelihoods_normal = fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms=chi_squared_terms_normal, 
                                                                                   noise_terms=noise_terms_normal)
        
        _model_images_sensitive = fitting.blur_images_including_blurring_regions(
            images=tracer_sensitive._image_plane_images, 
            blurring_images=tracer_sensitive._image_plane_blurring_images, 
            convolvers=convolvers)
        
        _residuals_sensitive = fitting.residuals_from_datas_and_model_datas(datas=sensitivity_images, 
                                                                         model_datas=_model_images_sensitive),
        _chi_squareds_sensitive = fitting.chi_squareds_from_residuals_and_noise_maps(residuals=_residuals_sensitive, 
                                                                                  noise_maps=_noise_maps),
        chi_squared_terms_sensitive = fitting.chi_squared_terms_from_chi_squareds(chi_squareds=_chi_squareds_sensitive)
        noise_terms_sensitive = fitting.noise_terms_from_noise_maps(noise_maps=_noise_maps)
        
        likelihoods_sensitive = fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms=chi_squared_terms_sensitive, 
                                                                                   noise_terms=noise_terms_sensitive)
        
        return sum(likelihoods_sensitive) - sum(likelihoods_normal)

    @property
    def likelihood(self):
        return self.fit_sensitive.likelihood - self.fit_normal.likelihood


def sensitivity_images_from_lensing_images_and_tracer_sensitive(lensing_images, tracer_sensitive, noise_seed=-1,
                                                                add_noise=True):

    convolvers_image = list(map(lambda fit_image: fit_image.convolver_image, lensing_images))
    map_to_scaled_arrays = list(map(lambda fit_data: fit_data.grids.image.scaled_array_from_array_1d,
                                    lensing_images))

    _mock_arrays = fitting.blur_images_including_blurring_regions(images=tracer_sensitive._image_plane_images,
                                                                  blurring_images=tracer_sensitive._image_plane_blurring_images,
                                                                  convolvers=convolvers_image)

    if add_noise:
        _mock_arrays = add_poisson_noise_to_mock_arrays(_mock_arrays=_mock_arrays, lensing_images=lensing_images,
                                                        seed=noise_seed)

    mock_arrays = fitting.map_arrays_to_scaled_arrays(_arrays=_mock_arrays, map_to_scaled_arrays=map_to_scaled_arrays)

    image = im.Image(array=mock_arrays[0], pixel_scale=lensing_images[0].image.pixel_scale,
                     noise_map=lensing_images[0].image.noise_map, psf=lensing_images[0].image.psf)

    return [MockSensitivityImage(image=image, mask=lensing_images[0].mask,
                                 convolver_image=lensing_images[0].convolver_image, grids=lensing_images[0].grids)]

def add_poisson_noise_to_mock_arrays(_mock_arrays, lensing_images, seed=-1):

    _mock_arrays_with_sky = list(map(lambda _mock_array, lensing_image:
                                     _mock_array + lensing_image.background_sky_map,
                                     _mock_arrays, lensing_images))


    _mock_arrays_with_sky_and_noise = list(map(lambda _mock_array_with_sky, lensing_image :
                    _mock_array_with_sky + im.generate_poisson_noise(image=_mock_array_with_sky,
                                        effective_exposure_map=lensing_image.effective_exposure_map, seed=seed),
                    _mock_arrays_with_sky, lensing_images))

    return list(map(lambda _mock_array_with_sky_and_noise, lensing_image:
                           _mock_array_with_sky_and_noise - lensing_image.background_sky_map,
                           _mock_arrays_with_sky_and_noise, lensing_images))


class MockSensitivityImage(im.Image):

    def __new__(cls, image, mask, convolver_image, grids):
        return np.array(mask.map_2d_array_to_masked_1d_array(image)).view(cls)

    def __init__(self, image, mask, convolver_image, grids):

        super().__init__(array=image, pixel_scale=image.pixel_scale,
                         noise_map=mask.map_2d_array_to_masked_1d_array(image.noise_map), psf=image.psf,
                         background_noise_map=image.background_noise_map)

        self.image = image
        self.mask = mask
        self.convolver_image = convolver_image
        self.grids = grids