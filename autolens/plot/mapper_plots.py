from autoarray.plot import plotters
from autoastro.plot import lensing_plotters


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_labels
def subplot_image_and_mapper(
    image,
    mapper,
    image_positions=None,
    source_positions=None,
    critical_curves=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=plotters.Include(),
    sub_plotter=None,
):

    number_subplots = 2

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    sub_plotter.plot_array(
        array=image,
        mask=include.mask_from_grid(grid=mapper.grid),
        positions=image_positions,
        lines=critical_curves,
        include_origin=include.origin,
    )

    if image_pixel_indexes is not None:

        sub_plotter.index_scatterer.scatter_grid_indexes(
            grid=image.mask.geometry.masked_grid, indexes=image_pixel_indexes
        )

    if source_pixel_indexes is not None:

        indexes = mapper.image_pixel_indexes_from_source_pixel_indexes(
            source_pixel_indexes=source_pixel_indexes
        )

        sub_plotter.index_scatterer.scatter_grid_indexes(
            grid=image.mask.geometry.masked_grid, indexes=indexes
        )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    sub_plotter.plot_mapper(
        mapper=mapper,
        positions=source_positions,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include.origin,
        include_grid=include.inversion_grid,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_border=include.inversion_border,
    )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.figure.close()
