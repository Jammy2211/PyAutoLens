import autofit as af
import autolens as al
import autolens.plot as aplt
import os

# This tool allows one to make simulated datasets of strong lenses, which can be used to test example pipelines and
# investigate strong lens modeling where the 'true' answer is known.

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the dataset is stored in, e.g:

# The image will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/image.fits'.
# The noise map will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/lens_name/noise_map.fits'.
# The psf will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/psf.fits'.

# Setup the path to the autolens_workspace, using a relative directory name.
plot_path = "{}/../images/simulating/".format(
    os.path.dirname(os.path.realpath(__file__))
)

# (these files are already in the autolens_workspace and are remade running this script)
dataset_name = "lens_sersic_sie__source_sersic"

# Create the path where the dataset will be output, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=plot_path, folder_names=[dataset_name]
)

# The grid use to create the image.
grid = al.Grid.uniform(shape_2d=(170, 170), pixel_scales=0.05, sub_size=4)

# Simulate a simple Gaussian PSF for the image.
psf = al.Kernel.from_gaussian(
    shape_2d=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

# To simulate the imaging dataset we first create a simulator, which defines the shape, resolution and pixel-scale of the
# image that is simulated, as well as its expoosure time, noise levels and psf.
simulator = al.SimulatorImaging(
    exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
    psf=psf,
    background_sky_map=al.Array.full(fill_value=0.1, shape_2d=grid.shape_2d),
    add_noise=True,
)

# Setup the lens galaxy's light (elliptical Sersic), mass (SIE+Shear) and source galaxy light (elliptical Sersic) for
# this simulated lens.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=45.0,
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=60.0,
        intensity=0.6,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


# Use these galaxies to setup a tracer, which will generate the image for the simulated imaging dataset.
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# Lets look at the tracer's image - this is the image we'll be simulating.

# To make this figure, we need to pass the plotter a grid which it uses to create the image. The simulator has its
# grid accessible as a property, which we can use to do this.
aplt.Tracer.profile_image(tracer=tracer, grid=grid)

# We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
# imaging dataset.
imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

# Lets plot the simulated imaging dataset before we output it to fits.
aplt.Imaging.subplot_imaging(imaging=imaging)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    overwrite=True,
)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Image"),
    output=aplt.Output(path=dataset_path, filename="image", format="png"),
)

aplt.Imaging.image(imaging=imaging, plotter=plotter)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Noise-Map"),
    output=aplt.Output(path=dataset_path, filename="noise_map", format="png"),
)

aplt.Imaging.noise_map(imaging=imaging, plotter=plotter)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="PSF"),
    output=aplt.Output(path=dataset_path, filename="psf", format="png"),
)

aplt.Imaging.psf(imaging=imaging, plotter=plotter)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.0
)
