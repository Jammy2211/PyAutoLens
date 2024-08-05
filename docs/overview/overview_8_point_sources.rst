.. _overview_8_point_sources:

Point Sources
=============

So far, overview examples have shown strongly lensed galaxies, whose extended surface brightness is lensed into
the awe-inspiring giant arcs and Einstein rings we see in high quality telescope imaging. There are many lenses where
the background source is not extended but is instead a point-source, for example strongly lensed quasars and supernovae.

For these objects, we do not want to model the source using light profiles, which implicitly assume an extended
surface brightness distribution. Instead, we assume that the source is a point source with a centre (y,x).

Our ray-tracing calculations no longer trace extended light rays from the source plane to the image-plane, but
instead now find the locations the point-source's multiple images appear in the image-plane.

Finding the multiple images of a mass model given a (y,x) coordinate in the source plane is an iterative problem
performed in a very different way to ray-tracing a light profile. In this example, we introduce **PyAutoLens**`s
`PositionSolver`, which does exactly this and thus makes the analysis of strong lensed quasars, supernovae and
point-like source's possible in **PyAutoLens**!

Source Plane Chi Squared
------------------------

This example performs point-source modeling using a source-plane chi-squared. This means the likelihood of a model
is evaluated based on how close the multiple image positions it predicts trace to the centre of the point source
in the source-plane.

This is often regard as a less robust way to perform point-source modeling than an image-plane chi-squared, and it
means that other information about the multiple images of the point source (e.g. their fluxes) cannot be used. On
the plus side, it is much faster to perform modeling using a source-plane chi-squared.

Visualization of point-source modeling results are also limited, as the feature is still in development.

Overview Script Quality
-----------------------

This script is less well written than others, lacking visualization, a coherent description and general writing
quality. This is because it is a work-in-progress and I am still finding time to present this feature in a clear
and concise way.

If you have a desire to use point source lens modeling, and are okay with a source-plane chi-squared analysis, I
recommend you go to the following modeling script which is more complete and better written:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/point_source/modeling/start_here.ipynb

Image Plane Chi Squared (In Development)
----------------------------------------

An image-plane chi-squared is also available, however it is an in development feature with limitations. The main
limitation is that the solver for the image-plane positions of a point-source in the source-plane is not robust. It
often infers incorrect additional multiple image positions or fails to locate the correct ones.

This is because I made the foolish decision to try and locate the positions by ray-tracing squares surrounding the
image-plane positions to the source-plane and using a nearest neighbor based approach based on the Euclidean distance.
This contrasts standard implementations elsewhere in the literature, which use a more robust approach based on ray
tracing triangles to the source-plane and using whether the source-plane position lands within each triangle.

This will one day be fixed, but we have so far not found time to do so.

Lensed Point Source
-------------------

To begin, we will create an image of strong lens using a simple isothermal mass model and source with an
exponential light profile.

Although we are going to show how **PyAutoLens**`s positional analysis tools model point-sources, showing the tools
using an extended source will make it visibly clearer where the multiple images of the point source are!

.. code-block:: python

    import autolens as al
    import autolens.plot as aplt

    grid = al.Grid2D.uniform(
        shape_native=(100, 100),
        pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
    )

    isothermal_mass_profile = al.mp.Isothermal(
        centre=(0.001, 0.001), einstein_radius=1.0, ell_comps=(0.0, 0.111111)
    )

    exponential_light_profile = al.lp.Exponential(
        centre=(0.07, 0.07),
        ell_comps=(0.2, 0.0),
        intensity=0.05,
        effective_radius=0.2,
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=isothermal_mass_profile)

    source_galaxy = al.Galaxy(redshift=1.0, light=exponential_light_profile)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

We plot the image of our strongly lensed source galaxy.

By eye, we can clearly see there are four multiple images located in a cross configuration, which are the
four (y,x) multiple image coordinates we want our positional solver to find!

.. code-block:: python

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

Here is the image:

[Missing]

Point Source
------------

The image above visually illustrates where the source's light traces in the image-plane.

Lets now treat this source as a point source, by setting up a source galaxy using the `Point` class.

.. code-block:: python

    point_source = al.ps.PointSourceChi(centre=(0.07, 0.07))

    source_galaxy = al.Galaxy(redshift=1.0, point_0=point_source)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

Position Solving
----------------

For a point source, our goal is to find the (y,x) coordinates in the image-plane that directly map to the centre
of the point source in the source plane. In this example, we therefore need to compute the 4 image-plane that map
directly to the location (0.07", 0.07"), the centre of the `Point` object above, in the source plane.

This is often referred to as 'solving the lens equation' in the literature.

This is an iterative problem that requires us to use the `PointSolver`.

.. code-block:: python

    solver = al.PointSolver(
        grid=grid,
        pixel_scale_precision=0.001,
        upscale_factor=2,
        distance_to_source_centre=0.01,
    )


We now pass the tracer to the solver. This will then find the image-plane coordinates that map directly to the
source-plane coordinate (0.07", 0.07"), which we plot below.

.. code-block:: python

    positions = solver.solve(lensing_obj=tracer, source_plane_coordinate=(0.07, 0.07))

    grid_plotter = aplt.Grid2DPlotter(grid=positions)
    grid_plotter.figure_2d()

Here is what the solved positions look like, compared to the observe data. In this example, the data was simulated
with the mass-model we used above, so the match is good:

[Missing]

You might be wondering why don't we use the image of the lensed source to compute our multiple images. Can`t we just
find the pixels in the image whose flux is brighter than its neighboring pixels?

Although this might work, for positional modeling we want to know the (y,x) coordinates of the multiple images at a
significantly higher precision than the grid we see the image on. In this example, the grid has a pixel scale of 0.05",
however we can determine our multiple image positions at scales of 0.01" or below!

Lens Modeling
-------------

**PyAutoLens** fully supports modeling strong lens datasets as a point-source. This might be used for analysing
strongly lensed quasars or supernovae, which are so compact we do not observe their extended emission.

To perform point-source modeling, we first create a `PointDataset` containing the image-plane (y,x) positions
of each multiple image and their noise values (which would be the resolution of the imaging data they are observed).

The positions below correspond to those of an isothermal mass model.

.. code-block:: python

    point_dataset = al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular(
            [[1.1488, -1.1488], [1.109, 1.109], [-1.109, -1.109], [-1.1488, 1.1488]]
        ),
        positions_noise_map=al.ArrayIrregular([0.05, 0.05, 0.05, 0.05]),
    )

Point Source Dictionary
-----------------------

In this simple example we model a single point source, which might correspond to one lensed quasar or supernovae.
However, **PyAutoLens** supports model-fits to datasets with many lensed point-sources, for example in galaxy clusters.

Each point source dataset is therefore passed into a `PointDict` object before the model-fit is performed. For
this simple example only one dataset is passed in, but in the galaxy-cluster examples you'll see this object makes it
straightforward to model datasets with many lensed sources.

.. code-block:: python

    point_dict = al.PointDict(point_dataset_list=[point_dataset])

We can print the ``positions`` of this dictionary and dataset, as well as their noise-map values.

.. code-block:: python

    print("Point Source Dataset Name:")
    print(point_dict["point_0"].name)
    print("Point Source Multiple Image (y,x) Arc-second Coordinates:")
    print(point_dict["point_0"].positions.in_list)
    print("Point Source Multiple Image Noise-map Values:")
    print(point_dict["point_0"].positions_noise_map.in_list)

Name Pairing
------------

Every point-source dataset in the `PointDict` has a name, which in this example was `point_0`. This `name` pairs
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in
the `PointDataset` **PyAutoLens** will raise an error.

In this example, where there is just one source, name pairing appears pointless. However, point-source datasets may
have many source galaxies in them, and name pairing is necessary to ensure every point source in the lens model is
fitted to its particular lensed images in the `PointDict`!

Fitting
-------

Just like we used a `Tracer` to fit imaging and interferometer data, we can use it to fit point-source data via the
`FitPoint` object.

This uses the names of each point-source in the dataset and model to create individual fits to the `positions`,
`fluxes` and other attributes that could be fitted. This allows us to inspect the residual-map,
chi-squared, likelihood, etc of every individual fit to part of our point source dataset.

.. code-block:: python

    fit = al.FitPointDict(point_dict=point_dict, tracer=tracer, point_solver=solver)

    print(fit["point_0"].positions.residual_map)
    print(fit["point_0"].positions.normalized_residual_map)
    print(fit["point_0"].positions.chi_squared_map)
    print(fit["point_0"].positions.log_likelihood)

Lens Modeling
-------------

It is straight forward to fit a lens model to a point source dataset, using the same API that we saw for dataset and
interferometer datasets.

This uses an ``AnalysisPoint`` object which fits the lens model in the correct way for a point source dataset.
This includes mapping the ``name``'s of each dataset in the ``PointDict`` to the names of the point sources in
the lens model.

.. code-block:: python

    # Lens:

    bulge = af.Model(al.lp.Sersic)
    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=bulge, mass=mass
    )

    # Source:

    point_0 = af.Model(al.ps.Point)

    source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

    # Overall Lens Model:

    galaxies = af.Collection(lens=lens, source=source)
    model = af.Collection(galaxies=galaxies)

    # Search + Analysis + Model-Fit

    search = af.Nautilus(name="overview_point_source")

    analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

    result = search.fit(model=model, analysis=analysis)

Wrap-Up
-------

The `point_source <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/point_source>`_ package of the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_  contains numerous example scripts for performing point source
modeling to datasets where there are only a couple of lenses and lensed sources, which fall under the category of
'galaxy scale' objects.

This also includes examples of how to add and fit other information that are observed by a point-source source,
for example the flux of each image.