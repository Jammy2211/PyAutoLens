"""
Census O3: ``import autolens`` must not pay for ``matplotlib.pyplot``.

``autolens/__init__.py`` imports the ``potential_correction`` sub-package,
whose ``__init__`` imports ``visualize``. That module used to import
``matplotlib.pyplot`` (and ``cm``/``ticker``/``mpl_toolkits``) at module level,
so every process that touched ``autolens`` at all paid ~0.32s to build a
pyplot state machine that only its six drawing functions use. Those imports now
live inside the functions, matching ``potential_correction/mesh.py`` and
``iterative.py``.
"""
import subprocess
import sys

from autolens.potential_correction import visualize


def test__importing_autolens_does_not_import_pyplot():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import autolens, sys; print('matplotlib.pyplot' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test__visualize_module_binds_no_matplotlib_names_at_module_level():
    # The deferral is only real while these stay off the module: a name put back
    # at the top of ``visualize.py`` would re-import pyplot on ``import autolens``
    # without failing any drawing test.
    for name in ("plt", "mpl", "cm", "ticker", "MaxNLocator", "make_axes_locatable"):
        assert not hasattr(visualize, name)


def test__drawing_functions_are_still_exported():
    for name in (
        "imshow_masked_data",
        "show_image_irregular_interpolate",
        "show_image_irregular",
        "show_fit_dpsi",
        "show_fit_dpsi_src",
    ):
        assert callable(getattr(visualize, name))
