Configs
=======

**PyAutoLens** uses a number of configuration files that customize the default behaviour of the non-linear searches,
visualization and other aspects of **PyAutoLens**. Here, we describe how to configure **PyAutoLens** to use the configs
and describe every configuration file complete with input parameters.

Setup
-----

By default, **PyAutoLens** looks for the config files in a ``config`` folder in the current working directory, which is
why we run autolens scripts from the ``autolens_workspace`` directory.

The configuration path can also be set manually in a script using the project **PyAutoConf** and the following
command (the path to the ``output`` folder where the results of a non-linear search are stored is also set below):

.. code-block:: bash

    from autoconf import conf

    conf.instance.push(
        config_path="path/to/config",
        output_path=f"path/to/output"
    )

general.ini
-----------

This config file is found at 'autolens_workspace/config/general.ini' and contains the following sections and variables:

[output]
    log_to_file
        If True the outputs of processes like the non-linear search are logged to a file (and not printed to screen).
    log_file
        The file name the logged output is written to (in the non-linear search output folder).
    log_level
        The level of logging.
    model_results_decimal_places
        The number of decimal places the estimated values and errors of all parameters in the model.results file are
        output to.
    remove_files
        If True, all output files of a non-linear search (e.g. samples, samples_backup, model.results, images, etc.)
        are deleted once the model-fit has completed.

        A .zip file of all output is always created before files are removed, thus results are not lost with this
        option turned on. If PyAutoLens does not find the output files of a model-fit (because they were removed) but
        does find this .zip file, it will unzip the contents and continue the analysis as if the files were
        there all along.

        This feature was implemented because super-computers often have a limit on the number of files allowed per
        user and the large number of files output by PyAutoLens can exceed this limit. By removing files the
        number of files is restricted only to the .zip files.
    force_pickle_overwrite
        A model-fit outputs pickled files of the model, search, results, etc., which the database feature can load.
        If this setting it ``True`` these pickle files are recreated when a new model-fit is performed, even if
        the search is complete.

The following setting flips all images that are loaded by **PyAutoLens** so that they appear the same orinetation as
the software ds9:

[fits]
    flip_for_ds9
        If ``True``, the ndarray of all .fits files containing an image, noise-map, psf, etc, is flipped upside down
        so its orientation is the same as ds9.

The following settings are specific for High Performance Super computer use with **PyAutoLens**.

[hpc]
    hpc_mode
        If ``True``, HPC mode is activated, which disables GUI visualization, logging to screen and other settings which
        are not suited to running on a super computer.
    iterations_per_update
        The number of iterations between every update (visualization, results output, etc) in HPC mode, which may be
        better suited to being less frequent on a super computer.

The following settings customize how a model is handled by **PyAutoFit**:

[model]
    ignore_prior_limits
        If ``True`` the limits applied to priors will be ignored, where limits set upper / lower limits. This should be
        disabled if one has difficult manipulating results in the database due to a ``PriorLimitException``.

The library `numba <https://github.com/numba/numba>`_ is used to speed up functions, by converting them to C callable
functions before the Python interpreter runs:

[numba]
    nopython
        If True, functions which hve a numba decorator must not convert back to Python during a run and thus must stay
        100% C. All PyAutoLens functions were developed with nopython mode turned on.
    cache
        If True, the C version of numba functions are cached on the hard-disk so they do not need to be
        recompiled every time **PyAutoLens** is rerun. If False, the first time every function is run will have a small
        delay (0.1-1.0 seconds) as it has to be numba compiled again.
    parallel
        If True, all functions decorated with the numba.jit are run with parallel processing on.

[inversion]
    interpolated_grid_shape {image_grid, source_grid}
        In order to output inversion reconstructions (which could be on a Voronoi grid) to a .fits file, the
        reconstruction is interpolated to a square grid of pixels. This option determines this grid:

        image_grid: The interpolated grid is the same shape, resolution and centering as the observed image-data.

        source_grid: The interpolated grid is zoomed to over-lay the source-plane reconstructed source and uses
        dimensions derived from the number of pixels used by the reconstruction.

[hyper]
    hyper_minimum_percent : float
        When creating hyper-images (see howtolens/chapter_5) all flux values below a certain value are rounded up an input
        value. This prevents negative flux values negatively impacting hyper-mode features or zeros creating division
        by zero errors.

        The value pixels are rounded to are the maximum flux value in the hyper image multipled by an input percentage
        value.

        The minimum percentage value the hyper image is mulitpled by in order to determine the value fluxes are rounded
        up to.
    hyper_noise_limit : float
        When noise scaling is activated (E.g. via hyper galaxies) this value is the highest value a noise value can
        numerically be scaled up too. This prevents extremely large noise map values creating numerically unstable
        log likelihood values.
    stochastic_outputs
        If ``True``, information on the stochastic likelihood behaviour of any KMeans based pixelization is output.

[test]
    test_mode
        If ``True`` this disables sampling of a search to provide a solution in one iteration. It is used for testing
        **PyAutoLens**.


non_linear
----------

The 'autolens_workspace/config/non_linear' config files contain the default settings used by every non-linear search.
The [search], [settings] and [initialize] sections of the non-linear configs contains settings specific to a
non-linear searches, and the documentation for these variables should be found by inspecting the
`API Documentation <https://pyautolens.readthedocs.io/en/latest/api/api.html>`_ of the relevent non-linear search
object.

The following config sections and variables are generic across all non-linear search configs:

[updates]
   iterations_per_update
        The number of iterations of the non-linear search performed between every 'update', where an update performs
        visualization of the maximum log likelihood model, backing-up of the samples, output of the model.results
        file and logging.
   visualize_every_update
        For every visualize_every_update updates visualization is performed and output to the hard-disk during the
        non-linear using the maximum log likelihood model. A visualization_interval of -1 turns off on-the-fly
        visualization.
   backup_every_update
        For every backup_every_update the results of the non-linear search in the samples foler and backed up into the
        samples_backup folder. A backup_every_update of -1 turns off backups during the non-linear search (it is still
        performed when the non-linear search terminates).
   model_results_every_update
        For every model_results_every_update the model.results file is updated with the maximum log likelihood model
        and parameter estimates with errors at 1 an 3 sigma confidence. A model_results_every_update of -1 turns off
        the model.results file being updated during the model-fit (it is still performed when the non-linear search
        terminates).
   log_every_update
        For every log_every_update the log file is updated with the output of the Python interpreter. A
        log_every_update of -1 turns off logging during the model-fit.

[printing]
    silence
        If True, the default print output of the non-linear search is silcened and not printed by the Python
        interpreter.

[prior_passer]
    sigma
        For non-linear search chaining and model prior passing, the sigma value of the inferred model parameter used
        as the sigma of the passed Gaussian prior.
    use_errors
        If ``True``, the errors of the previous model's results are used when passing priors.
    use_widths
        If ``True`` the width of the model parameters defined in the priors config file are used.

[parallel]
    number_of_cores
        For non-linear searches that support parallel procesing via the Python multiprocesing module, the number of
        cores the parallel run uses. If number_of_cores=1, the model-fit is performed in serial omitting the use
        of the multi-processing module.

The output path of every non-linear search is also 'tagged' using strings based on the [search] setting of the
non-linear search:

visualize
---------

These config files are found at 'autolens_workspace/config/visualize' and they contain the default settings used by
visualization in **PyAutoLens**. The majority of config files are described in the ``autolens_workspace/plot`` package.

The *general.ini* config contains the following sections and variables:

[general]
    backend
        The matploblib backend used for visualization (see
        https://gist.github.com/CMCDragonkai/4e9464d9f32f5893d837f3de2c43daa4 for a description of backends).

        If you use an invalid backend for your computer, **PyAutoLens** may crash without an error or reset your machine.
        The following backends have worked for **PyAutoLens** users:

        TKAgg (default)

        Qt5Agg (works on new MACS)

        Qt4Agg

        WXAgg

        WX

        Agg (outputs to .fits / .png but doesn't'display figures during a run on your computer screen)

The ``plots.ini`` config file customizes every image that is output to hard-disk during a model-fit.

The ``include.ini`` config file customizes every feature that appears on plotted images by default (e.g. crtiical
curves, a mask, light profile centres, etc.).

priors
------

These config files are found at 'autolens_workspace/config/priors' and they contain the default priors and related
variables for every light profile and mass profile when it is used as a model. They appear as follows:

.. code-block:: bash

    "SphIsothermal": {
        "einstein_radius": {
            "type": "Uniform",
            "lower_limit": 0.0,
            "upper_limit": 4.0,
            "width_modifier": {
                "type": "Absolute",
                "value": 0.2
            },
            "gaussian_limits": {
                "lower": 0.0,
                "upper": "inf"
            }
        }

The sections of this example config set the following:

json config
    type {Uniform, Gaussian, LogUniform}
        The default prior given to this parameter when used by the non-linear search. In the example above, a
        UniformPrior is used with lower_limit of 0.0 and upper_limit of 4.0. A GaussianPrior could be used by
        putting "Gaussian" in the "type" box, with "mean" and "sigma" used to set the default values. Any prior can be
        set in an analogous fashion (see the example configs).
    width_modifier
        When the results of a search are passed to a subsequent search to set up the priors of its non-linear search,
        this entry describes how the Prior is passed. For a full description of prior passing, checkout the examples
        in 'autolens_workspace/examples/complex/linking'.
    gaussian_limits
        When the results of a search are passed to a subsequent search, they are passed using a GaussianPrior. The
        gaussian_limits set the physical lower and upper limits of this GaussianPrior, such that parameter samples
        can not go beyond these limits.

notation
--------

The notation configs define the labels of every model parameter and its derived quantities, which are used when 
visualizing results (for example labeling the axis of the PDF triangle plots output by a non-linear search).

Two examples using the 1D data fitting example for the config file **label.ini** are:

[label]
    centre_0
        The label given to that parameter for non-linear search plots using that parameter, e.g. cornerplot PDF plots.
        For example, if centre_1=x, the plot axis will be labeled 'x'.

[superscript]
    EllIsothermal
        The superscript used on certain plots that show the results of different model-components. For example, if
        EllIsothermal=mass, plots where the EllIsothermal are plotted will have a superscript `mass`.

The **label_format.ini** config file specifies the format certain parameters are output as in output files like the
*model.results* file. This uses standard Python formatting strings.

grids
-----

**radial_minimum.ini**

The calculation of many quantities from light profiles and mass profiles, for example their image, convergence
or deflection angles are ill-defined at (y,x) coordinates (0.0, 0.0). This can lead **PyAutoLens** to crash if not
handled carefully.

The *radial_minimum.ini* config file defines, for every profile, the values coordinates at (0.0, 0.0) are rounded to
to prevent these numerical issues. For example, if the value of a profile is 1e-8, than input coordinates of (0.0, 0.0)
will be rounded to values (1e-8, 0.0).