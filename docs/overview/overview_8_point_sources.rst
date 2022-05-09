.. _overview_7_point_sources:

Point Sources
=============

So far, we have shown strongly lensed galaxies whose extended surface brightness is lensed into the awe-inspiring
giant arcs and Einstein rings we see in high quality telescope imaging. There are many lenses where the background
source is not extended but is instead a point-source, for example strongly lensed quasars and supernovae.

For these objects, we do not want to model the source using a ``LightProfile`` which implicitly assumes an extended
surface brightness distribution. Instead, we assume that our source is a point source with a centre (y,x). Our
ray-tracing calculations no longer trace extended light rays from the source plane to the image-plane, but instead
now find the locations the point-source's multiple images appear in the image-plane.

Here is an example of a compact source that has been simulated in **PyAutoLens**, with the positions of its four
multiple images marked using stars:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/point_sources/image.png
  :width: 400
  :alt: Alternative text

Point Sources
-------------

For point source modeling, our goal is to find the multiple images of a lens mass model given a (y,x) coordinate in the
source plane. This is an iterative problem performed in a very different way to ray-tracing used when evaluating a
``LightProfile``.

To treat a source as a point source, we create it as a galaxy using ``Point`` object and pass it to a tracer:

.. code-block:: python

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllIsothermal(
            centre=(0.001, 0.001), einstein_radius=1.0, elliptical_comps=(0.0, 0.111111)
        )
    )

    point_source = al.ps.Point(centre=(0.07, 0.07))

    source_galaxy = al.Galaxy(redshift=1.0, point_0=point_source)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

Position Solving
----------------

This tracer allows us to compute deflection angles and therefore map image-pixels to the source-plane. We can therefore
pass it to the ``PositionSolver`` object to solve for the image-pixel coordinates which correspond to the source's
source-plane position at (0.07", 0.07").

There are many options that can be passed to a ``PositionsSovler``, below we only specify the precision we want the
image-plane coordinates to be calculated too, which is 0.001", much lower than the resolution fo the data we observed
the point-sources on!

.. code-block:: python

    solver = al.PointSolver(
        grid=grid_2d,
        pixel_scale_precision=0.001,
    )

    positions = solver.solve(lensing_obj=tracer, source_plane_coordinate=(0.07, 0.07))

Here is what the solved positions look like, compared to the observe data. In this example, the data was simulated
with the mass-model we used above, so the match is good:

[Missing]

Point Source Dataset
--------------------

**PyAutoLens** has full support for analysing strong lens datasetsas a point-source. This might be used for analysing
strongly lensed quasars or supernovae, which are so compact we do not observe their extended emission.

To perform point-source analysing, we first create a ``PointDataset`` containing the image-plane (y,x) positions
of each multiple image and their noise values (which would be the resolution of the imaging data they are observed).

The positions below correspond to those of an ``EllIsothermal`` mass model.

.. code-block:: python

    point_dataset = al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular(
            [[1.1488, -1.1488], [1.109, 1.109], [-1.109, -1.109], [-1.1488, 1.1488]]
        ),
        positions_noise_map=al.ValuesIrregular([0.05, 0.05, 0.05, 0.05]),
    )

In this simple example we model a single point source, which might correspond to one lensed quasar or supernovae.
However, **PyAutoLens** supports model-fits to datasets with many lensed point-sources, which is used for analysing
group-scale and cluster-scale strong lenses.

Each point source dataset is therefore passed into a ``PointDict`` object before the model-fit is performed. For
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

Every point-source dataset in the ``PointDict`` has a name, which in this example was ``point_0``. This ``name``
pairs the dataset to the ``Point`` in the model below. Because the name of the dataset is ``point_0``, the
only ``Point`` object that is used to fit it must have the name ``point_0``.

This ensures if a dataset has many point sources (e.g. galaxy clusters) it is clear how the model pairs the data.

Fitting
-------

Just like we used a ``Tracer`` to fit imaging and interferometer data, we can use it to
fit point-source data via the ``FitPoint`` object.

This uses the names of each point-source in the dataset and model to create individual fits to the ``positions``,
``fluxes`` and other attributes that could be fitted. This allows us to inspect the residual-map,
chi-squared, likelihood, etc of every individual fit to part of our point source dataset.

.. code-block:: python

    fit = al.FitPointDict(point_dict=point_dict, tracer=tracer, point_solver=solver)

    print(fit["point_0"].positions.residual_map)
    print(fit["point_0"].positions.chi_squared_map)
    print(fit["point_0"].positions.log_likelihood)

Lens Modeling
-------------

It is straight forward to fit a lens model to a point source dataset, using the same API that we saw for imaging and
interferometer datasets.

This uses an ``AnalysisPoint`` object which fits the lens model in the correct way for a point source dataset.
This includes mapping the ``name``'s of each dataset in the ``PointDict`` to the names of the point sources in
the lens model.

.. code-block:: python

    lens_galaxy_model = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal)
    source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.Point)

    galaxies = af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)
    model = af.Collection(galaxies=galaxies)

    search = af.DynestyStatic(name="overview_point_source")

    analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

    result = search.fit(model=model, analysis=analysis)

Wrap-Up
-------

The `point_source <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/point_source>`_ package of the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_  contains numerous example scripts for performing point source
modeling to datasets where there are only a couple of lenses and lensed sources, which fall under the category of
'galaxy scale' objects.

This also includes examples of how to add and fit other information that are observed by a point-source source,
for example the flux of each image.