from autolens.visualize import array_plotters

def plot_observed_image_from_image(image, normalization='linear', norm_min=None, norm_max=None,
                                   output_path=None, output_filename=None, output_type='show'):

    array_plotters.plot_observed_image_array(array=image, xticks=image.xticks, yticks=image.yticks,
                                             normalization=normalization, norm_min=norm_min, norm_max=norm_max,
                                             output_path=output_path, output_filename=output_filename,
                                             output_type=output_type)

def plot_residuals_from_fitter(fitter, normalization='symmetric_log', norm_min=None, norm_max=None,
                               output_path=None, output_filename=None, output_type='show'):

    array_plotters.plot_residuals_array(array=fitter.residuals,
                                        xticks=fitter.lensing_image.image.xticks, yticks=fitter.lensing_image.image.yticks,
                                        normalization=normalization, norm_min=norm_min, norm_max=norm_max,
                                        output_path=output_path, output_filename=output_filename, output_type=output_type)

def plot_chi_squareds_from_fitter(fitter, normalization='symmetric_log', norm_min=None, norm_max=None,
                               output_path=None, output_filename=None, output_type='show'):

    array_plotters.plot_chi_squareds_array(array=fitter.chi_squareds,
                                        xticks=fitter.lensing_image.image.xticks, yticks=fitter.lensing_image.image.yticks,
                                        normalization=normalization, norm_min=norm_min, norm_max=norm_max,
                                        output_path=output_path, output_filename=output_filename, output_type=output_type)