# from autolens.inversion import mappers
#
# def plot_reconstruction(mapper, inversion,
#                         points, grid, as_subplot,
#                         units, kpc_per_arcsec,
#                         xyticksize,
#                         norm, norm_min, norm_max, linthresh, linscale,
#                         figsize, aspect, cmap, cb_ticksize,
#                         title, titlesize, xlabelsize, ylabelsize,
#                         output_path, output_filename, output_format):
#
#     if isinstance(mapper, mappers.RectangularMapper):
#
#         plot_rectangular_reconstruction(mapper, inversion,
#                                         points, grid, as_subplot,
#                                         units, kpc_per_arcsec,
#                                         xyticksize,
#                                         norm, norm_min, norm_max, linthresh, linscale,
#                                         figsize, aspect, cmap, cb_ticksize,
#                                         title, titlesize, xlabelsize, ylabelsize,
#                                         output_path, output_filename, output_format)
#
# def plot_rectangular_reconstruction(mapper, inversion,
#                                     points, grid, as_subplot,
#                                     units, kpc_per_arcsec, xyticksize,
#                                     norm, norm_min, norm_max, linthresh, linscale,
#                                     figsize, aspect, cmap, cb_ticksize,
#                                     title, titlesize, xlabelsize, ylabelsize,
#                                     output_path, output_filename, output_format):
#
#     reconstructed_pixelization = mapper.reconstructed_pixelization_from_solution_vector(inversion.solution_vector)
#
#     array_plotters.plot_array(array=reconstructed_pixelization, points=points, grid=grid, as_subplot=as_subplot,
#             units=units, kpc_per_arcsec=kpc_per_arcsec,
#             xticks=reconstructed_pixelization.xticks, yticks=reconstructed_pixelization.yticks, xyticksize=xyticksize,
#             norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
#             figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
#             title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
#             output_path=output_path, output_filename=output_filename, output_format=output_format)