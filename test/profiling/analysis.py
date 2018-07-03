from src.analysis import analysis as an
from src.analysis import galaxy as g
from src.profiles import light_profiles, mass_profiles
from src.imaging import image as im
from src.imaging import mask as msk
from src.imaging import scaled_array
import os

repeats = 10

# Load up the data
lens_name = 'source_sersic'
data_dir = "../../data/" + lens_name.format(os.path.dirname(os.path.realpath(__file__)))

data = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/image', hdu=0, pixel_scale=0.1)
noise = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/noise', hdu=0, pixel_scale=0.1)
exposure_time = scaled_array.ScaledArray.from_fits(file_path=data_dir + '/exposure_time', hdu=0,
                                                   pixel_scale=0.1)
psf = im.PSF.from_fits(file_path=data_dir + '/psf', hdu=0, pixel_scale=0.1)

image = im.Image(array=data, effective_exposure_time=exposure_time, pixel_scale=0.1, psf=psf,
                 background_noise=noise, poisson_noise=noise)

mask = msk.Mask.circular(shape_arc_seconds=image.shape_arc_seconds, pixel_scale=data.pixel_scale, radius_mask=2.0)


def repeat(func):
    print("rpe")
    for _ in range(repeats):
        print("repeat")
        func(_)


def test_analysis_1():
    analysis = an.Analysis(image, mask)

    source_galaxy = g.Galaxy(light_profile=light_profiles.EllipticalSersic())
    lens_galaxy = g.Galaxy(spherical_mass_profile=mass_profiles.EllipticalIsothermal(axis_ratio=0.9),
                           shear_mass_profile=mass_profiles.ExternalShear())

    repeat(lambda _: analysis.run(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy]))


if __name__ == "__main__":
    test_analysis_1()
