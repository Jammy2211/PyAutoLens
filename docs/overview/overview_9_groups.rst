.. _overview_8_groups:

Group-Scale Lenses
==================

The strong lenses we've discussed so far have just a single lens galaxy responsible for the lensing, with a single
source galaxy observed to be lensed. A strong lensing group is a system where there are multiple lens galaxies,
deflecting the one or more background sources:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/groups/image.png
  :width: 400
  :alt: Alternative text

The Source's ring is much larger than other examples (> 5.0") and there are clearly additional galaxies in and around
the main lens galaxy. 

Point Source
------------

Modeling group scale lenses is challenging, because each individual galaxy must be included in the overall lens model. 
For this simple overview, we will therefore model the system as a point source, which reduces the complexity of the 
model and reduces the computational run-time of the model-fit.

.. code-block:: python

    point_dict = al.PointDict.from_json(
        file_path=path.join(dataset_path, "point_dict.json")
    )

    point_solver = al.PointSolver(grid=grid_2d, pixel_scale_precision=0.025)

Model via JSON
--------------

We now compose the lens model. For groups there could be many hundreds of galaxies in the model. Whereas previous
examples explicitly wrote the model out via Python code, for group modeling we opt to write the model in .json files,
which is loaded as follows:

.. code-block:: python

    model_path = path.join("scripts", "group", "models")
    model_file = path.join(model_path, "lens_x3__source_x1.json")

    lenses_file = path.join(model_path, "lenses.json")
    lenses = af.Collection.from_json(file=lenses_file)

    sources_file = path.join(model_path, "sources.json")
    sources = af.Collection.from_json(file=sources_file)

    galaxies = lenses + sources

    model = af.Collection(galaxies=galaxies)

This .json file contains all the information on this particular lens's model, including priors which adjust their
centre to the centre of light of each lens galaxy. The script used to make the model can be viewed at
the `following link <https://github.com/Jammy2211/autolens_workspace/blob/master/scripts/group/model_maker/lens_x3__source_x1.py>`_.

Lens Modeling
-------------

We are now able to model this dataset as a point source, using the exact same tools we used in the point source
overview.

.. code-block:: python

    search = af.DynestyStatic(name="overview_groups")

    analysis = al.AnalysisPoint(
        point_dict=point_dict, solver=point_solver
    )

    result = search.fit(model=model, analysis=analysis)

Result
------

The result contains information on every galaxy in our lens model:

.. code-block:: python

    print(result.max_log_likelihood_instance.galaxies.lens_0.mass)
    print(result.max_log_likelihood_instance.galaxies.lens_1.mass)
    print(result.max_log_likelihood_instance.galaxies.lens_2.mass)

Extended Source Fitting
-----------------------

For group-scale lenses like this one, with a modest number of lens and source galaxies, **PyAutoLens** has all the
tools you need to perform extended surface-brightness fitting to the source's extended emission, including the use
of a pixelized source reconstruction.

This will extract a lot more information from the data than the point-source model and the source reconstruction means
that you can study the properties of the highly magnified source galaxy. Here is what the fit looks like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/groups/fit_group.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/groups/source_group.png
  :width: 400
  :alt: Alternative text

This type of modeling uses a lot of **PyAutoLens**'s advanced model-fitting features which are described in chapters 3
and 4 of the **HowToLens** tutorials. An example performing this analysis to the lens above can be found
at `this link. <https://github.com/Jammy2211/autolens_workspace/blob/master/notebooks/group/chaining/point_source_to_imaging.ipynb>`_

Wrap-Up
-------

The `group <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/group>`_ package of the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ contains numerous example scripts for performing group-sale modeling
and simulating group-scale strong lens datasets.