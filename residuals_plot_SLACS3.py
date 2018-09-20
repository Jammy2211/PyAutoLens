from autolens.imaging import scaled_array
from autolens.visualize import array_plotters

new_shape = (200, 200)
xticks = [-3.0, -1.5, 1.5, 3.0]
yticks = [-3.0, -1.5, 1.5, 3.0]

obs = scaled_array.ScaledArray.from_fits(file_path='/gpfs/data/pdtw24/PL_Data/Image_Maker/SLACS_03_OddPSF/slacs_3_post.fits',
                                         hdu=1)

obs = obs.trim(new_shape)

array_plotters.plot_image(image=obs, xticks=xticks, yticks=yticks, normalization='symmetric_log',
                          output_path='/home/jammy/Documents/', output_filename='SLACS1250_Obs', output_format='png')

residuals_one_sersic = scaled_array.ScaledArray.from_fits(file_path='/home/jammy/Documents/RichardImages/Residuals_fits/'
                                                                    'SLACSJ1250+0523_Decomp_Mass_x1Sersic.fits', hdu=0)

residuals_one_sersic = residuals_one_sersic.trim(new_shape)

array_plotters.plot_residuals(residuals=residuals_one_sersic, xticks=xticks, yticks=yticks, norm_min=-0.1, norm_max=0.1,
                              output_path='/home/jammy/Documents/', output_filename='SLACS1250_x1Sersic', output_format='png')

# stop

residuals_one_sersic = scaled_array.ScaledArray.from_fits(file_path='/home/jammy/Documents/RichardImages/Residuals_fits/'
                                                                    'SLACSJ1250+0523_Decomp_Mass_x2SersicOffset.fits', hdu=0)

residuals_one_sersic = residuals_one_sersic.trim(new_shape)

array_plotters.plot_residuals(residuals=residuals_one_sersic, xticks=xticks, yticks=yticks, norm_min=-0.1, norm_max=0.1,
                              output_path='/home/jammy/Documents/', output_filename='SLACS1250_x2Sersic', output_format='png')