from autolens.imaging import scaled_array
from autolens.visualize import array_plotters

new_shape = (100, 100)
xticks = [-1.5, -0.5, 0.5, 1.5]
yticks = [-1.5, -0.5, 0.5, 1.5]

obs = scaled_array.ScaledArray.from_fits(file_path='/gpfs/data/pdtw24/PL_Data/Image_Maker/SLACS_03_OddPSF/slacs_5_post.fits',
                                         hdu=1)

obs = obs.trim(new_shape)

array_plotters.plot_observed_image_array(array=obs, xticks=xticks, yticks=yticks, output_path='/home/jammy/Documents/',
                                         normalization='symmetric_log', output_filename='SLACS0252_Obs', output_type='png')

residuals_1_sersic = scaled_array.ScaledArray.from_fits(file_path='/home/jammy/Documents/RichardImages/Residuals_fits/'
                                                        'SLACSJ0252+0039_Decomp_Mass_x1Sersic.fits', hdu=0)

residuals_1_sersic = residuals_1_sersic.trim(new_shape)

array_plotters.plot_residuals_array(array=residuals_1_sersic, xticks=xticks, yticks=yticks, norm_min=-0.05, norm_max=0.1,
                                    output_path='/home/jammy/Documents/', output_filename='SLACS0252_x1Sersic', output_type='png')


residuals_2_sersic = scaled_array.ScaledArray.from_fits(file_path='/home/jammy/Documents/RichardImages/Residuals_fits/'
                                                        'SLACSJ0252+0039_Decomp_Mass_x2SersicOffset.fits', hdu=0)

residuals_2_sersic = residuals_2_sersic.trim(new_shape)

array_plotters.plot_residuals_array(array=residuals_2_sersic, xticks=xticks, yticks=yticks, norm_min=-0.05, norm_max=0.1,
                                    output_path='/home/jammy/Documents/',
                                    output_filename='SLACS0252_x2Sersic', output_type='png')