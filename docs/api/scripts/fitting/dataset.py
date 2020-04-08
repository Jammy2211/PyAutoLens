import autolens as al
import autolens.plot as aplt
import os

# Setup the path to the autolens_workspace, using a relative directory name.
plot_path = "{}/../images/fitting/".format(os.path.dirname(os.path.realpath(__file__)))
dataset_path = "{}/dataset/".format(os.path.dirname(os.path.realpath(__file__)))

# Using the dataset path, load the data (image, noise-map, PSF) as an imaging object from .fits files.
imaging = al.Imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=0.05,
)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Image"),
    output=aplt.Output(path=plot_path, filename="image", format="png"),
)

aplt.Imaging.image(imaging=imaging, plotter=plotter)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Noise-Map"),
    output=aplt.Output(path=plot_path, filename="noise_map", format="png"),
)

aplt.Imaging.noise_map(imaging=imaging, plotter=plotter)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="PSF"),
    output=aplt.Output(path=plot_path, filename="psf", format="png"),
)

aplt.Imaging.psf(imaging=imaging, plotter=plotter)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.0
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Image"),
    output=aplt.Output(path=plot_path, filename="masked_image", format="png"),
)

aplt.Imaging.image(imaging=masked_imaging, mask=mask, plotter=plotter)
