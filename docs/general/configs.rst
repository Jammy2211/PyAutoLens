Configs
=======

The autolens workspace includes a set of configuration files that customize the behaviour of the non-linear searches,
visualization and other aspects of **PyAutoLens**. Here, we describe how to configure **PyAutoLens** to use the configs
and describe every configuration file complete with input parameters.

Setup
-----

By default, **PyAutoLens** looks for the config files in a ``config`` folder in the current working directory, which is
why we run autolens scripts from the ``autolens_workspace`` directory.

The configuration path can also be set manually in a script using **PyAutoConf** and the following command (the path
to the ``output`` folder where the results of a ``NonLinearSearch`` are stored is also set below):

.. code-block:: bash

    from autoconf import conf

    conf.instance.push(
        config_path="path/to/config", output_path=f"path/to/output"
    )

general.ini
-----------

This config file is found at 'autolens_workspace/config/general.ini' and contains the following sectioons and variables:

[output]
    log_file -> str
        The file name the logged output is written to (in the `NonLinearSearch` output folder).
    log_level -> str
        The level of logging.
    model_results_decimal_places -> int
        The number of decimal places the estimated values and errors of all parameters in the model.results file are
        output to.
    remove_files -> bool
        If True, all output files of a `NonLinearSearch` (e.g. samples, samples_backup, model.results, images, etc.)
        are deleted once the model-fit has completed.

        A .zip file of all output is always created before files are removed, thus results are not lost with this
        option turned on. If PyAutoLens does not find the output files of a model-fit (because they were removed) but
        does find this .zip file, it will unzip the contents and continue the analysis as if the files were
        there all along.

        This feature was implemented because super-computers often have a limit on the number of files allowed per
        user and the large number of files output by PyAutoLens can exceed this limit. By removing files the
        number of files is restricted only to the .zip files.
    skip_completed -> bool
        If True, and if the results of a `NonLinearSearch` were completed in a previous run, then all processing steps
        performed at the end of the `NonLinearSearch` (e.g. output of sample results, visualization, etc.) are skipped.

        If False, they are repeated, which can be used for updating visualization or the `NonLinearSearch` pickles
        to a new version of PyAutoLens.
    grid_results_interval -> int
        For a GridSearch non-linear optimization this interval sets after how many samples on the grid output is
        performed for. A grid_results_interval of -1 turns off output.

The library `numba <https://github.com/numba/numba>`_ is used to speed up functions, by converting them to C callable
functions before the Python interpreter runs:

[numba]
    nopython -> bool
        If True, functions which hve a numba decorator must not convert back to Python during a run and thus must stay
        100% C. All PyAutoLens functions were developed with nopython mode turned on.
    cache -> bool
        If True, the C version of numba-fied functions are cached on the hard-disk so they do not need to be
        re-numbafied every time PyAutoLens is rerun. If False, the first time every function is run will have a small
        delay (0.1-1.0 seconds) as it has to be numba-fied again.
    parallel -> bool
        If True, all functions decorated with the numba.jit are run with parallel processing on.

The calculation grid is used to compute lensing quantities that do not inherently need a grid to be passed to them
for them to be calculated. For example, the critical curves and caustics of a mass model should be computed in
a way that does not need a grid to be input.

The calculation grid is the homogenous grid used to calculate all lensing quantities of this nature. It is computed
as follows:

1) Draw a rectangular 'bounding box' around the `MassProfile`'s convergence profile, where the four side of the
   the box are at threshold values of convergence.

2) Use a grid of this box to compute the desired lensing quantity (e.g. the critical curve).

In a future version of PyAutoLens the calculation grid will be adaptive, such that the values input into this config
file are the desired precision of the quantitiy being calculated (e.g. the area of the critical curve should not
change as the grid resolution is increased within a threshold value). Unfortunately, we've not yet had time
to implement this adaptive grid.

[calculation_grid]
    convergence_threshold -> float
        The threshold value of convergence at which the 4 sides of the bounding box described above are located.
    pixels -> int
        The shape_2d of the grid inside the bounding box from which the lensing quantitiy is computed (e.g. it is shape
        (pixels, pixels)).

[inversion]
    interpolated_grid_shape -> str {image_grid, source_grid}
        In order to output inversion reconstructions (which could be on a Voronoi grid) to a .fits file, the
        reconstruction is interpolated to a square grid of pixels. This option determines this grid:

        image_grid - The interpolated grid is the same shape, resolution and centering as the observed image-data.

        source_grid - The interpolated grid is zoomed to over-lay the source-plane reconstructed source and uses
        dimensions derived from the number of pixels used by the reconstruction.
    inversion_pixel_limit_overall -> int
        The maximum number of pixels that may be assumed for an inversion during a `NonLinearSearch` fit.

[hyper]
    hyper_minimum_percent -> float
        When creating hyper-images (see howtolens/chapter_5) all flux values below a certain value are rounded up an input
        value. This prevents negative flux values negatively impacting hyper-mode features or zeros creating division
        by zero errors.

        The value pixels are rounded to are the maximum flux value in the hyper image multipled by an input percentage
        value.

        The minimum percentage value the hyper image is mulitpled by in order to determine the value fluxes are rounded
        up to.

non_linear
----------

These config files are found at 'autolens_workspace/config/non_linear' and they contain the default settings used by
every non-linear search. The [search], [settings] and [initialize] sections of the non-linear configs contains settings
specific to certain non-linear searches, and the documentation for these variables should be found by inspecting the
`API Documentation <https://pyautolens.readthedocs.io/en/latest/api/api.html>`_ of the relevent `NonLinearSearch` object.

The following config sections and variables are generic across all `NonLinearSearch` configs (e.g.
config/non_linear/nest/DynestyStatic.ini, config/non_linear/mcmc/Emcee.ini, etc.):

[updates]
   iterations_per_update -> int
        The number of iterations of the `NonLinearSearch` performed between every 'update', where an update performs
        visualization of the maximum log likelihood model, backing-up of the samples, output of the model.results
        file and logging.
   visualize_every_update -> int
        For every visualize_every_update updates visualization is performed and output to the hard-disk during the
        non-linear using the maximum log likelihood model. A visualization_interval of -1 turns off on-the-fly
        visualization.
   backup_every_update -> int
        For every backup_every_update the results of the `NonLinearSearch` in the samples foler and backed up into the
        samples_backup folder. A backup_every_update of -1 turns off backups during the `NonLinearSearch` (it is still
        performed when the `NonLinearSearch` terminates).
   model_results_every_update -> int
        For every model_results_every_update the model.results file is updated with the maximum log likelihood model
        and parameter estimates with errors at 1 an 3 sigma confidence. A model_results_every_update of -1 turns off
        the model.results file being updated during the model-fit (it is still performed when the non-linear search
        terminates).
   log_every_update -> int
        For every log_every_update the log file is updated with the output of the Python interpreter. A
        log_every_update of -1 turns off logging during the model-fit.

[printing]
    silence -> bool
        If True, the default print output of the `NonLinearSearch` is silcened and not printed by the Python
        interpreter.

[prior_passer]
sigma=3.0
use_errors=True
use_widths=True

[parallel]
    number_of_cores -> int
        For non-linear searches that support parallel procesing via the Python multiprocesing module, the number of
        cores the parallel run uses. If number_of_cores=1, the model-fit is performed in serial omitting the use
        of the multi-processing module.

The output path of every `NonLinearSearch` is also 'tagged' using strings based on the [search] setting of the
non-linear search:

[tag]
    name -> str
        The name of the `NonLinearSearch` used to start the tag path of output results. For example for the non-linear
        search DynestyStatic the default name tag is 'dynesty_static'.

visualize
---------

These config files are found at 'autolens_workspace/config/visualize' and they contain the default settings used by
visualization in **PyAutoLens**. The *general.ini* config contains the following sections and variables:

[general]
    backend -> str
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

priors
------

These config files are found at 'autolens_workspace/config/priors' and they contain the default priors and related
variables for every model-component in a project, using .json format files (as opposed to .ini. for most config files).

The autolens workspace contains example json_prior files for the 1D ``data`` fitting problem. An example entry of the
json configs for the ``sigma`` parameter of the ``Gaussian`` class is as follows:

.. code-block:: bash

    "Gaussian": {
        "sigma": {
            "type": "Uniform",
            "lower_limit": 0.0,
            "upper_limit": 30.0,
            "width_modifier": {
                "type": "Absolute",
                "value": 0.2
            },
            "gaussian_limits": {
                "lower": 0.0,
                "upper": "inf"
            }
        },

The sections of this example config set the following:

json config
    type -> Prior
        The default prior given to this parameter which is used by the `NonLinearSearch`. In the example above, a
        UniformPrior is used with lower_limit of 0.0 and upper_limit of 30.0. A GaussianPrior could be used by
        putting "Gaussian" in the "type" box, with "mean" and "sigma" used to set the default values. Any prior can be
        set in an analogous fashion (see the example configs).
    width_modifier
        When the results of a phase are linked to a subsequent phase to set up the priors of its non-linear search,
        this entry describes how the Prior is passed. For a full description of prior passing, checkout the examples
        in 'autolens_workspace/examples/complex/linking'.
    gaussian_limits
        When the results of a phase are linked to a subsequent phase, they are passed using a GaussianPrior. The
        gaussian_limits set the physical lower and upper limits of this GaussianPrior, such that parameter samples
        can not go beyond these limits.

notation
--------

The notation configs define the labels of every model-component parameter and its derived quantities, which are
used when visualizing results (for example labeling the axis of the PDF triangle plots output by a non-linear search).
Two examples using the 1D ``data`` fitting example for the config file **label.ini** are:

[label]
    centre_0 -> str
        The label given to that parameter for `NonLinearSearch` plots using that parameter, e.g. the PDF plots. For
        example, if centre_1=x, the plot axis will be labeled 'x'.

[subscript]
    EllipticalIsothermal -> str
        The subscript used on certain plots that show the results of different model-components. For example, if
        EllipticalIsothermal=m, plots where the EllipticalIsothermal are plotted will have a subscript m.

The **label_format.ini** config file specifies the format certain parameters are output as in output files like the
*model.results* file.

The **tags.ini** config file specifies the tag of every `SettingsPhase`, *SetupPipeline* and *SLaM* input variable,
where these tags customize the output path of the `NonLinearSearch` in a unique way based on how the model-fitting
procedure is set up.

Tags are self-explanatory and named after the input value of the class they are paired with. For a description of the
settings themselves checkout the `API Documentation <https://pyautolens.readthedocs.io/en/latest/api/api.html>`_.

grids
-----

**interpolate.ini**

The `GridInterpolate` class speeds up the calculation of lensing quantities such as the potential or deflection angles
by computing them on a grid of reduced resolution and interpolating the results to a grid at the native resolution of
the data. This is important for certain mass profiles, where the calculations require computationally expensive
numerical integration.

The *interpolate.ini* specifies for every `LightProfile` and `MassProfile` in **PyAutoLens** whether, when a
`GridInterpolate` object is passed into a from grid method (e.g deflections_from_grid) the calculation should be
performed using interpolation or by computing every value on the grid explicitly at native resolution.

The default *interpolate.ini* config file supplied with the **PyAutoLens** workspace specifies `False` for every
profile that does not require numerical integration (and therefore is fast to compute) and `True` for every profile
which does (and therefore can see the calculation sped ups by factors of > x10).

**radial_minimum.ini**

The calculation of many quantities from `LightProfile`'s and *MassProfile's*, for example their image, convergence
or deflection angles are ill-defined at (y,x) coordinates (0.0, 0.0). This can lead **PyAutoLens** to crash if not
handled carefully.

The *radial_minimum.ini* config file defines, for every profile, the values coordinates at (0.0, 0.0) are rounded to
to prevent these numerical issues. For example, if the value of a profile is 1e-8, than input coordinates of (0.0, 0.0)
will be rounded to values (1e-8, 0.0).