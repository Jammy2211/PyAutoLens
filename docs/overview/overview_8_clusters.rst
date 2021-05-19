.. _overview_8_clusters:

Clusters
--------

The strong lenses we've discussed so far have just a single lens galaxy responsible for the lensing, with a single
source galaxy observed to be lensed. A strong lensing cluster is a system where there are multiple lens galaxies,
deflecting the one or more background sources:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/clusters/cluster.png
  :width: 400
  :alt: Alternative text

Galaxy clusters range in scale, between the following two extremes:

 - Groups: Strong lenses with a distinct 'primary' lens galaxy and a handful of lower mass galaxies nearby. These
 typically have just one or two lensed sources whose arcs are visible. The Einstein Radii of these systems typically
range from range 5.0" -> 10.0" (with galaxy scale lenses typically below 5.0").

 - Clusters: These are objects with tens or hundreds of lens galaxies and lensed sources, where the lensed sources
 are all at different redshfits. .

**PyAutoLens** has tools for modeling cluster datasets anywhere between these two extremes. Lets first consider a
group scale lens:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/clusters/cluster.png
  :width: 400
  :alt: Alternative text

The Source's ring is much larger than other examples (> 5.0") and there are clearly additional galaxies in and around
the main lens galaxy.

Modeling group scale lenses is challenging, because each individual galaxy must be included in the overall lens model.
For this simple overview, we will therefore model the system as a point source, which reduces the complexity of the
model and reduces the computational run-time of the model-fit.

.. code-block:: bash

    point_source_dict = al.PointSourceDict.from_json(
        file_path=path.join(dataset_path, "point_source_dict.json")
    )

    positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.025)

We now compose the lens model. For clusters there could be many hundreds of galaxies in the model. Whereas previous
examples explicitly wrote the model out via Python code, for cluster modeling we opt to write it in .json files which
are loaded in this script, as follows:

.. code-block:: bash

    model_file = path.join("path", "to", "lens_x3__source_x1.json")
    model = af.Collection.from_json(file=model_file)

This .json file contains all the information on this particular lens's model, including priors which adjust their
centre to the centre of light of each lens galaxy. The file can be viewed at
the `following link <https://github.com/Jammy2211/autolens_workspace/blob/master/scripts/clusters/modeling/models/lens_x3__source_x1.json>`_.

We are now able to model this dataset as a point source, using the exact same tools we used in the point source
overview.

.. code-block:: bash

    search_1 = af.DynestyStatic(name="overview_clusters_group")

    analysis = al.AnalysisPointSource(
        point_source_dict=point_source_dict, solver=positions_solver
    )

    result_1 = search_1.fit(model=model, analysis=analysis)

For group-scale lenses like this one, with a modest number of lens and source galaxies, **PyAutoLens** has all the
tools you need to perform extended surface-brightness fitting to the source's extended emission, including the use
of a pixelized source reconstruction.

This will extract a lot more information from the data than the point-source model and the source reconstruction means
that you can study the properties of the highly magnified source galaxy. Here is what the fit looks like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/clusters/fit_group.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/clusters/source_group.png
  :width: 400
  :alt: Alternative text

This type of modeling uses a lot of **PyAutoLens**'s advanced model-fitting features which are described in chapters 3
and 4 of the **HowToLens** tutorials. An example performing this analysis to the lens above can be found
at `this link <https://github.com/Jammy2211/autolens_workspace/blob/master/notebooks/clusters/chaining/point_source_to_imaging.ipynb>`_