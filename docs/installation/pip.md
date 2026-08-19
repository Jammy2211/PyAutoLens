(pip)=

# Installation with pip

:::{note}
**PyAutoLens** requires **Python 3.12 or later**. On Python 3.9, 3.10 or 3.11,
`pip install autolens` stops with an error telling you to upgrade — it will not
quietly install an older release instead. Upgrade Python to 3.12+ before
installing.
:::

## JAX & GPU

**PyAutoLens** runs significantly faster on GPUs — often **50x or more** compared to CPUs.

This acceleration is achieved through \[**JAX**\](<https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>), which provides GPU and TPU support.

**JAX is installed by default** — a plain `pip install autolens` includes it (the older
`pip install autolens[jax]` command still works and installs the same thing).

The default install includes **CPU-only** JAX. To ensure GPU acceleration, it is recommended that you install JAX with
GPU support **before** installing **PyAutoLens**, by following the official \[JAX installation guide\](<https://jax.readthedocs.io/en/latest/installation.html>).

If you install **PyAutoLens** without a proper GPU setup, a warning will be displayed.

:::{note}
**Intel Macs**: JAX no longer publishes wheels for Intel (x86_64) macOS, so on these machines
`pip install autolens` automatically installs without JAX and runs on the slower NumPy path — a
warning is printed at import to make this clear. Every other supported platform (Windows, Linux,
Apple-silicon Macs) gets JAX by default.
:::

## Install

We strongly recommend that you install **PyAutoLens** in a
[Python virtual environment](https://www.geeksforgeeks.org/python-virtual-environment/), with the link attached
describing what a virtual environment is and how to create one.

We upgrade pip to ensure certain libraries install:

```bash
pip install --upgrade pip
```

The latest version of **PyAutoLens** is installed via pip as follows:

```bash
pip install autolens
```

This includes JAX by default, enabling the acceleration described above. If you need an install without
JAX on a platform where JAX wheels exist (e.g. a restricted environment), install normally and then remove it:

```bash
pip uninstall jax jaxlib
```

**PyAutoLens** detects the absence at import and falls back to the fully supported (but much slower)
NumPy path.

If pip prints warnings about dependency version conflicts, these can usually be ignored — the instructions below
will identify clearly if the installation is a success.

If there are no errors **PyAutoLens** is installed!

If there is an error check out the [troubleshooting section](https://pyautolens.readthedocs.io/en/latest/installation/troubleshooting.html).

## Workspace

Next, clone the `autolens workspace` (the line `--depth 1` clones only the most recent branch on
the `autolens_workspace`, reducing the download size):

```bash
cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
git clone https://github.com/PyAutoLabs/autolens_workspace --depth 1
cd autolens_workspace
```

Run the `welcome.py` script to get started!

```bash
python3 welcome.py
```

It should be clear that **PyAutoLens** runs without issue.

If there is an error check out the [troubleshooting section](https://pyautolens.readthedocs.io/en/latest/installation/troubleshooting.html).

## Numba

Numba (<https://numba.pydata.org>) is an optional library which makes **PyAutoLens** run a lot faster, which we
strongly recommend users have installed.

You can install numba via the following command:

```bash
pip install numba
```

Some users have experienced difficulties installing numba, which is why it is an optional library. If your
installation is not successful, you can use **PyAutoLens** without it installed for now, to familiarize yourself
with the software and determine if it is the right software for you.

If you decide that **PyAutoLens** is the right software, then I recommend you commit the time to getting a
successful numba install working, with more information provided [at this readthedocs page](https://pyautolens.readthedocs.io/en/latest/installation/numba.html)

## Optional

For interferometer analysis there are two optional dependencies that must be installed via the commands:

```bash
pip install pynufft
```

**PyAutoLens** will run without these libraries and it is recommended that you only install them if you intend to
do interferometer analysis.

If you run interferometer code a message explaining that you need to install these libraries will be printed, therefore
it is safe not to install them initially.

## Legacy Python versions

We dropped support for Python 3.9, 3.10 and 3.11 in release `2026.7.29.2`
(July 2026) — the first release published declaring `Requires-Python >=3.12`.

Raising that floor does not retract what is already published. Releases at or
below `2026.7.29.1` were published declaring `>=3.9`, and PyPI metadata is
immutable, so they remain valid candidates forever. Left alone, `pip install
autolens` on an older Python did not fail — it walked back to `2026.7.29.1` and
installed a months-old stack without JAX, reporting nothing.

Release `2026.7.29.1.post1` exists to stop that. It contains no code, declares
`Requires-Python <3.12`, and raises an error when pip tries to build it, so an
unsupported Python gets an explanation instead of a stale install.

If you need a historical release, pin it exactly — that still resolves on older
Pythons:

```bash
pip install autolens==2025.10.6.1
```

One gap remains: `pip install --only-binary=:all: autolens` skips source
distributions entirely, so it steps past `2026.7.29.1.post1` and installs the
old wheel silently. If you use that flag, pin the version you want.
