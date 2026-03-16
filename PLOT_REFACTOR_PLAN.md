# Plot Module Refactoring Plan

Remove `MatPlot`, `MatWrap`, `Visuals`, and `Output` from PyAutoArray/PyAutoGalaxy/PyAutoLens
in favour of direct matplotlib calls with explicit parameters.

---

## Goals

- Delete `MatPlot1D`, `MatPlot2D` and all `~50` `MatWrap` wrapper classes
- Delete `Visuals1D`, `Visuals2D` and all subclasses
- Delete `Output` — replaced by a `save_figure()` helper function
- Delete all `mat_wrap*.yaml` config files — only `visualize/general.yaml` (figsize) survives
- Keep all `*Plotter` classes as the public API (internal wiring rewired)
- Keep `plots.yaml` which controls which subplots are auto-generated during analysis runs
- All unit tests pass after every PR

---

## What is wrong with the current design

### MatWrap / MatPlot

Every matplotlib concept (colormap, ticks, colorbar, title, axis extent, …) has a
corresponding Python class that loads default values from a YAML config file.
There are ~50 such classes, each with a `figure:` and `subplot:` config section,
totalling three config files (~10 KB of YAML) just for plot defaults.

The indirection adds no value: the same result is achieved with plain function
default-parameter values, which are visible in the code and require no config lookup.

### Visuals

`Visuals2D` is a dataclass of overlays (critical curves, caustics, centres, …).
It has many variants to satisfy the config-switching machinery. The same information
can be passed as typed list/array arguments to the plot functions.

### The subplot state machine

The current system tracks subplot position through a mutable integer
`mat_plot_2d.subplot_index` that auto-increments after every plot call.

Problems:
- Developers manually patch it (`self.mat_plot_2d.subplot_index = 6`) to skip slots
- `mat_plot_1d` and `mat_plot_2d` have *independent* counters; a comment in the code
  describes the workaround as a "nasty hack"
- The config system switches every wrap object between `figure:` and `subplot:` sections
  based on whether `subplot_index is not None`, adding hidden state to every config lookup
- Nested plotters (FitImagingPlotter → TracerPlotter → InversionPlotter) share one
  mat_plot object so their indices accumulate in the same global counter

The fix is to use matplotlib's native `plt.subplots()` and pass `ax` objects directly.

### Ticks

`XTicks` / `YTicks` MatWrap classes have special-case logic for log-scale ticks.
The replacement generates 3 evenly spaced linear ticks from the extent using
`np.linspace` (as in the reference `plot_grid()` example).  Log scales on colorbars
are handled by passing `LogNorm()` to `imshow` — matplotlib handles the ticks itself.

---

## New design in one picture

```
Plotter.subplot_fit()
  │
  ├── fig, axes = plt.subplots(3, 4, figsize=conf_figsize("subplots"))
  │
  ├── plot_array(array=fit.data,           title="Data",     ax=axes[0,0])
  ├── plot_array(array=fit.noise_map,      title="Noise",    ax=axes[0,1])
  ├── plot_array(array=fit.model_image,    title="Model",    ax=axes[0,2])
  │             lines=[critical_curves],
  ├── ...
  │
  ├── save_figure(fig, path=output_path, filename="subplot_fit")
  └── plt.close(fig)

Plotter.figure_convergence(ax=None)
  │
  ├── owns_figure = ax is None
  ├── if owns_figure: fig, ax = plt.subplots(1, 1, figsize=conf_figsize("figures"))
  │
  ├── plot_array(array=tracer.convergence, title="Convergence", ax=ax,
  │             lines=critical_curves + radial_curves)
  │
  └── if owns_figure: save_figure(fig, ...) ; plt.close(fig)
```

Key rules:
1. Every `plot_*` function accepts an optional `ax` parameter.
   - `ax=None` → creates its own figure, saves/shows, closes.
   - `ax` provided → draws onto it, does **not** save/show/close (caller is responsible).
2. Overlay data (critical curves, caustics, centres, positions, …) are plain
   `List[np.ndarray]` arguments, not `Visuals` objects.
3. `figsize` is the only value read from config; all other defaults are function
   parameter defaults visible in source code.
4. Ticks: 3 linear ticks generated with `np.linspace` from the axis extent.
   Colorbar log-scaling uses `matplotlib.colors.LogNorm` passed to `imshow`.

---

## `save_figure` replacing `Output`

```python
# autoarray/plot/plots/utils.py

def save_figure(
    fig: plt.Figure,
    path: str,
    filename: str,
    format: str = "png",
    dpi: int = 300,
) -> None:
    """Save fig to <path>/<filename>.<format> and close it."""
    os.makedirs(path, exist_ok=True)
    fig.savefig(
        os.path.join(path, f"{filename}.{format}"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)
```

The plotter base class holds `output_path: str` and `output_format: str = "png"`.
Individual `figure_*` and `subplot_*` methods call `save_figure(fig, self.output_path, "name")`.
If `output_path` is empty, `plt.show()` is called instead of saving.

---

## `conf_figsize` helper

```python
# autoarray/plot/plots/utils.py

def conf_figsize(context: str = "figures") -> Tuple[int, int]:
    """Read figsize from visualize/general.yaml for 'figures' or 'subplots'."""
    return tuple(conf.instance["visualize"]["general"][context]["figsize"])
```

`visualize/general.yaml` (only surviving config for plots):
```yaml
figures:
  figsize: [7, 7]
subplots:
  figsize: [19, 16]
```

---

## The 13 PRs

### Phase 1 — PyAutoArray (5 PRs)

---

#### PR A1 · New `autoarray/plot/plots/` module (additive, no deletions)

Create the replacement plot functions.  No existing code is touched; existing tests
continue to pass.

```
autoarray/plot/plots/
    __init__.py
    utils.py        → save_figure(), conf_figsize(), _make_ticks(), _apply_extent()
    array.py        → plot_array()
    grid.py         → plot_grid()
    yx.py           → plot_yx()
    inversion.py    → plot_inversion_reconstruction(), plot_inversion_mappings()
```

**`plot_array` signature (canonical example):**

```python
def plot_array(
    array: np.ndarray,
    ax: Optional[plt.Axes] = None,
    # overlays
    mask: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    positions: Optional[List[np.ndarray]] = None,
    lines: Optional[List[np.ndarray]] = None,
    vector_yx: Optional[np.ndarray] = None,
    # cosmetics
    title: str = "",
    xlabel: str = "x (arcsec)",
    ylabel: str = "y (arcsec)",
    colormap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_log10: bool = False,
    # figure control (used only when ax is None)
    figsize: Optional[Tuple[int, int]] = None,
    filename: Optional[str] = None,
) -> None:
    owns_figure = ax is None
    if owns_figure:
        figsize = figsize or conf_figsize("figures")
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    norm = LogNorm() if use_log10 else None
    if vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(array, cmap=colormap, norm=norm, origin="lower")
    plt.colorbar(im, ax=ax)

    if mask is not None:
        ax.scatter(mask[:, 1], mask[:, 0], s=1, c="k")
    if positions is not None:
        for pos in positions:
            ax.scatter(pos[:, 1], pos[:, 0], s=10, c="r")
    if lines is not None:
        for line in lines:
            ax.plot(line[:, 1], line[:, 0], linewidth=2)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)

    if owns_figure:
        if filename:
            save_figure(fig, path=os.path.dirname(filename),
                        filename=os.path.basename(filename))
        else:
            plt.show()
            plt.close(fig)
```

**Ticks:** The 3-linear-tick approach from the reference example is baked into
`_make_ticks(extent)`:
```python
def _apply_extent(ax, extent):
    """extent = [xmin, xmax, ymin, ymax]; apply axis limits and 3 linear ticks."""
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks(np.linspace(extent[0], extent[1], 3))
    ax.set_yticks(np.linspace(extent[2], extent[3], 3))
```
No `XTicks` / `YTicks` classes needed.  Log colorbars: pass `LogNorm()` to `imshow`;
matplotlib generates appropriate log-spaced colorbar ticks automatically.

New unit tests: `test_autoarray/plot/plots/test_array.py` etc., asserting that
PNG files are written when a filename is provided.

---

#### PR A2 · Update `Array2DPlotter` and `Grid2DPlotter`

Switch the two most-used base plotters to the new functions.

- Remove `mat_plot_2d`, `visuals_2d` constructor params.
- Add explicit overlay params: `mask`, `grid`, `positions`, `lines`.
- Each `figure_*` method calls `plot_array(..., ax=ax)` where `ax` defaults to `None`.
- `subplot_*` methods: create `fig, axes = plt.subplots(...)`, pass each `ax` slice.

The subplot open/close/index machinery is **deleted**.  A subplot method looks like:

```python
def subplot_array(self):
    fig, axes = plt.subplots(1, 2, figsize=conf_figsize("subplots"))
    self.figure_array(ax=axes[0])
    self.figure_array_log10(ax=axes[1])
    save_figure(fig, self.output_path, "subplot_array", self.output_format)
```

No `subplot_index`, no `open_subplot_figure()`, no `close_subplot_figure()`.

Existing test assertions about output filenames keep working because plotter
constructor accepts `output_path` and `output_filename` strings.

---

#### PR A3 · Update `ImagingPlotter`, `InversionPlotter`, `MapperPlotter`, `InterferometerPlotter`

Same `ax`-passing pattern.  Mixed 1D/2D subplots (e.g. interferometer) use:

```python
fig, axes = plt.subplots(2, 3, figsize=conf_figsize("subplots"))
plot_array(array=dirty_image, ax=axes[0, 0])
plot_yx(y=visibilities.real, ax=axes[1, 0])
```

`AbstractPlotter` base class is simplified to hold only:

```python
class AbstractPlotter:
    def __init__(
        self,
        output_path: str = "",
        output_filename: str = "",
        output_format: str = "png",
        figsize_figures: Optional[Tuple] = None,
        figsize_subplots: Optional[Tuple] = None,
    ):
        self.output_path = output_path
        self.output_filename = output_filename
        self.output_format = output_format
        self.figsize_figures = figsize_figures or conf_figsize("figures")
        self.figsize_subplots = figsize_subplots or conf_figsize("subplots")

    def _filename(self, name: str) -> Optional[str]:
        if self.output_path:
            return os.path.join(self.output_path,
                                f"{name}.{self.output_format}")
        return None
```

No subplot state, no mat_plot slots, no visuals slots.

---

#### PR A4 · Delete `mat_plot/`, `wrap/`, `visuals/` directories

```
autoarray/plot/mat_plot/    ← deleted (3 files)
autoarray/plot/wrap/        ← deleted (~40 files)
autoarray/plot/visuals/     ← deleted (3 files)
autoarray/config/visualize/mat_wrap.yaml      ← deleted
autoarray/config/visualize/mat_wrap_1d.yaml   ← deleted
autoarray/config/visualize/mat_wrap_2d.yaml   ← deleted
```

Update `autoarray/plot/__init__.py` to remove all `MatPlot*`, `Visuals*`, `MatWrap*`
re-exports.  Tests that imported these classes directly are deleted or rewritten.

---

#### PR A5 · Simplify config; finalise `save_figure` / `conf_figsize`

`visualize/general.yaml` after cleanup:

```yaml
figures:
  figsize: [7, 7]
subplots:
  figsize: [19, 16]
```

All other YAML files that existed purely for MatWrap defaults are deleted.
`plots.yaml` (which controls whether `subplot_fit` etc. are auto-generated during
analysis runs) is **kept unchanged**.

---

### Phase 2 — PyAutoGalaxy (4 PRs)

---

#### PR G1 · New galaxy overlay helpers (additive)

```
autogalaxy/plot/plots/
    __init__.py
    overlays.py     → overlay_critical_curves(ax, curves, color="w", linewidth=2)
                      overlay_caustics(ax, curves, color="y", linewidth=2)
                      overlay_light_profile_centres(ax, centres, marker="+", s=40)
                      overlay_mass_profile_centres(ax, centres, marker="x", s=40)
                      overlay_multiple_images(ax, positions, marker="o", s=40)
```

These are pure overlay helpers that accept an `ax` and draw onto it.
They have no config dependency.

---

#### PR G2 · Update `LightProfilePlotter`, `MassProfilePlotter`, `GalaxyPlotter`, `GalaxiesPlotter`

Each plotter computes its own overlay data from its galaxy/profile then passes it
to `plot_array`:

```python
class GalaxiesPlotter(AbstractPlotter):
    def figure_image(self, ax=None):
        owns = ax is None
        if owns:
            fig, ax = plt.subplots(figsize=self.figsize_figures)

        array = self.galaxies.image_2d_from(grid=self.grid)
        plot_array(array=array.native, ax=ax, title="Image",
                   lines=self._critical_curves() + self._caustics())
        _apply_extent(ax, self._extent())

        if owns:
            save_figure(fig, self.output_path, "image", self.output_format)
```

Remove autogalaxy `MatPlot2D` subclass and autogalaxy `Visuals2D` subclass.

---

#### PR G3 · Update autogalaxy `FitImagingPlotter` and `FitInterferometerPlotter`

```python
def subplot_fit(self):
    fig, axes = plt.subplots(3, 4, figsize=self.figsize_subplots)

    plot_array(array=self.fit.data.native,   title="Data",  ax=axes[0, 0])
    plot_array(array=self.fit.noise_map.native, title="Noise", ax=axes[0, 1])
    # ... etc., no subplot_index needed

    save_figure(fig, self.output_path, "subplot_fit", self.output_format)
```

---

#### PR G4 · Remove autogalaxy MatPlot/Visuals extensions

```
autogalaxy/plot/mat_plot/   ← deleted
autogalaxy/plot/visuals/    ← deleted
```

Update `autogalaxy/plot/__init__.py`.

---

### Phase 3 — PyAutoLens (4 PRs)

---

#### PR L1 · Update `TracerPlotter`

The plotter computes critical curves / caustics itself from the tracer, then passes
them as `lines` to `plot_array`:

```python
class TracerPlotter(AbstractPlotter):
    def figure_convergence(self, ax=None):
        owns = ax is None
        if owns:
            fig, ax = plt.subplots(figsize=self.figsize_figures)

        array = self.tracer.convergence_2d_from(self.grid)
        tang = self.tracer.tangential_critical_curves_from(self.grid)
        rad  = self.tracer.radial_critical_curves_from(self.grid)

        plot_array(array=array.native, ax=ax, title="Convergence",
                   lines=tang + rad)

        if owns:
            save_figure(fig, self.output_path,
                        self.output_filename or "convergence")

    def subplot_tracer(self):
        fig, axes = plt.subplots(3, 3, figsize=self.figsize_subplots)

        self.figure_image(ax=axes[0, 0])
        self.figure_source_plane(ax=axes[0, 1])
        self.figure_convergence(ax=axes[0, 2])
        self.figure_potential(ax=axes[1, 0])
        self.figure_magnification(ax=axes[1, 1])
        self.figure_deflections_y(ax=axes[1, 2])
        self.figure_deflections_x(ax=axes[2, 0])
        axes[2, 1].set_visible(False)
        axes[2, 2].set_visible(False)

        save_figure(fig, self.output_path, "subplot_tracer")
```

Constructor: remove `mat_plot_2d`, `visuals_2d`, `visuals_2d_of_planes_list`.
Add `show_critical_curves: bool = True`, `show_caustics: bool = True`.

---

#### PR L2 · Update `FitImagingPlotter`

Largest single plotter.  The 12-panel `subplot_fit` becomes:

```python
def subplot_fit(self):
    fig, axes = plt.subplots(3, 4, figsize=self.figsize_subplots)

    plot_array(array=self.fit.data.native,
               title="Data", ax=axes[0, 0])
    plot_array(array=self.fit.signal_to_noise_map.native,
               title="Signal-To-Noise Map", ax=axes[0, 1])
    plot_array(array=self.fit.model_image.native,
               title="Model Image", ax=axes[0, 2],
               lines=self._tangential_critical_curves())
    # leave axes[0, 3] blank or use for something else

    # plane decomposition (delegate to sub-plotter with explicit ax)
    tracer_plotter = self.tracer_plotter_of_plane(plane_index=0)
    tracer_plotter.figure_plane_image(ax=axes[1, 0])

    plot_array(array=self.fit.residual_map.native,
               title="Residual Map", ax=axes[2, 0],
               colormap="coolwarm", vmin=-0.1, vmax=0.1)
    plot_array(array=self.fit.normalized_residual_map.native,
               title="Normalised Residual Map", ax=axes[2, 1],
               colormap="coolwarm", vmin=-3, vmax=3)
    plot_array(array=self.fit.chi_squared_map.native,
               title="Chi-Squared Map", ax=axes[2, 2])

    save_figure(fig, self.output_path, "subplot_fit")
```

No `subplot_index`, no `open_subplot_figure`, no `close_subplot_figure`,
no 1D/2D sync.

Per-plane subplot (`subplot_of_planes`) creates its own figure:
```python
def subplot_of_planes(self):
    n = len(self.fit.tracer.planes)
    fig, axes = plt.subplots(1, n * 4, figsize=(n * 4 * 4, 4))
    for i in range(n):
        ...
```

---

#### PR L3 · Update `FitInterferometerPlotter`, `PointDatasetPlotter`, `FitPointDatasetPlotter`

**PointDatasetPlotter** — mixed 1D/2D, which was the "nasty hack" case:

```python
def subplot_dataset(self):
    fig, axes = plt.subplots(1, 2, figsize=self.figsize_subplots)

    plot_grid(grid=self.dataset.positions.array,
              y_errors=self.dataset.positions_noise_map.array,
              title="Positions", ax=axes[0])

    plot_yx(y=self.dataset.fluxes.array,
            y_errors=self.dataset.fluxes_noise_map.array,
            title="Fluxes", ax=axes[1])

    save_figure(fig, self.output_path, "subplot_dataset")
```

No sync hack: `axes[0]` is independent from `axes[1]`, they are just different `Axes`
objects obtained from the same `plt.subplots()` call.

---

#### PR L4 · Update `SubhaloPlotter`, `SubhaloSensitivityPlotter`; clean up `autolens/plot/abstract_plotters.py`

`autolens/plot/abstract_plotters.py` final form:

```python
from autogalaxy.plot.abstract_plotters import AbstractPlotter

class Plotter(AbstractPlotter):
    """PyAutoLens plotter base — no MatPlot or Visuals slots."""
    pass
```

SubhaloPlotter significance maps use `plot_array` with an `ArrayOverlay` equivalent:

```python
plot_array(
    array=self.result.figure_of_merit_array().native,
    title="Subhalo Detection Significance",
    ax=ax,
    positions=self.result.subhalo_centres_grid.array,
)
```

---

## Summary table

| PR | Repo | Change type | Tests |
|---|---|---|---|
| A1 | autoarray | Add `plots/` module | New unit tests |
| A2 | autoarray | Rewrite Array2D/Grid2DPlotter | Update existing |
| A3 | autoarray | Rewrite Imaging/Inversion/Mapper/InterferometerPlotter | Update existing |
| A4 | autoarray | Delete mat_plot/, wrap/, visuals/ | Delete wrap tests |
| A5 | autoarray | Config cleanup, finalise helpers | Smoke tests |
| G1 | autogalaxy | Add overlay helpers | New unit tests |
| G2 | autogalaxy | Rewrite Galaxy/Mass/LightProfile plotters | Update existing |
| G3 | autogalaxy | Rewrite FitImaging/FitInterferometer plotters | Update existing |
| G4 | autogalaxy | Delete MatPlot2D/Visuals2D extensions | Delete wrap tests |
| L1 | autolens | Rewrite TracerPlotter | Update existing |
| L2 | autolens | Rewrite FitImagingPlotter | Update existing |
| L3 | autolens | Rewrite FitInterferometer/Point plotters | Update existing |
| L4 | autolens | Rewrite Subhalo plotters, clean abstract_plotters | Update existing |

---

## Design rules applied consistently across all PRs

1. **`ax` parameter on every `figure_*` method and every `plot_*` function.**
   `ax=None` → owns the figure (creates, saves, closes).
   `ax` provided → draws only, caller owns the figure lifecycle.

2. **Overlay data as typed list/array args.**
   `lines: List[np.ndarray]` replaces `Visuals2D.tangential_critical_curves` etc.
   `positions: List[np.ndarray]` replaces `Visuals2D.positions`.
   No `Visuals` objects anywhere.

3. **No subplot state machine.**
   `plt.subplots(rows, cols)` returns `axes`; pass each `ax` slice explicitly.
   No `subplot_index`, no `open_subplot_figure`, no `close_subplot_figure`.
   Blank panels: `ax.set_visible(False)`.

4. **`figsize` from config only.** Every other default (fontsize, colormap, marker size,
   linewidth, …) is an inline function-parameter default, visible in source code.

5. **Linear ticks: `np.linspace(lo, hi, 3)`.**
   Log colorbars: pass `matplotlib.colors.LogNorm()` to `imshow`; matplotlib generates
   log-spaced colorbar ticks automatically — no custom tick class needed.

6. **`save_figure(fig, path, filename, format, dpi)` replaces `Output`.**
   If `output_path` is empty string, call `plt.show()` + `plt.close()` instead of saving.

7. **No deprecation warnings.** Old `mat_plot_2d` / `visuals_2d` constructor parameters
   are simply removed; callers are updated in the same PR.
