from autolens.visualize import util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

def plot_observed_image_array(array, xticks, yticks, normalization='log', norm_min=None, norm_max=None,
                              output_path=None, output_filename=None, output_type='show'):


    plt.figure(figsize=(20, 15))

    norm_min, norm_max = util.get_normalization_min_max(array, norm_min, norm_max)
    norm = util.get_normalization(normalization, norm_min, norm_max, linthresh=0.05, linscale=0.01)

    plt.imshow(array, aspect='auto', cmap='jet', norm=norm)
    util.set_title_and_labels(title='Observed Image', xlabel='x (arcsec)', ylabel='y (arcsec)')
    util.set_ticks(array=array, xticks=xticks, yticks=yticks)
    util.set_colorbar(norm_min, norm_max)
    util.output_array(array=array, output_path=output_path, output_filename=output_filename, output_type=output_type)

    plt.close()

def plot_residuals_array(array, xticks, yticks, normalization='symmetric_log', norm_min=None, norm_max=None,
                         output_path=None, output_filename=None, output_type='show'):

    plt.figure(figsize=(20, 15))

    norm_min, norm_max = util.get_normalization_min_max(array, norm_min, norm_max)
    norm = util.get_normalization(normalization, norm_min, norm_max, linthresh=0.001, linscale=0.001)

    plt.imshow(array, aspect='auto', cmap='jet', norm=norm)
    util.set_ticks(array=array, xticks=xticks, yticks=yticks)
    util.set_title_and_labels(title='Image Residuals', xlabel='x (arcsec)', ylabel='y (arcsec)')
    util.set_colorbar(norm_min, norm_max)
    util.output_array(array=array, output_path=output_path, output_filename=output_filename, output_type=output_type)

    plt.close()

def plot_chi_squareds_array(array, xticks, yticks, normalization='log', norm_min=None, norm_max=None,
                         output_path=None, output_filename=None, output_type='show'):

    plt.figure(figsize=(20, 15))

    norm_min, norm_max = util.get_normalization_min_max(array, norm_min, norm_max)
    norm = util.get_normalization(normalization, norm_min, norm_max, linthresh=0.001, linscale=0.001)

    plt.imshow(array, aspect='auto', cmap='jet', norm=norm)
    util.set_ticks(array=array, xticks=xticks, yticks=yticks)
    util.set_title_and_labels(title='Image Chi Squareds', xlabel='x (arcsec)', ylabel='y (arcsec)')
    util.set_colorbar(norm_min, norm_max)
    util.output_array(array=array, output_path=output_path, output_filename=output_filename, output_type=output_type)

    plt.close()