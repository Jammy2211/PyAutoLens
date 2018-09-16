from autolens.visualize import util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

def plot_observed_image_array(array, units, xticks, yticks, xyticksize=40,
                              norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                              figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                              title='Observed Image (electrons per second)', titlesize=46, xlabelsize=36, ylabelsize=36,
                              output_path=None, output_filename='observed_image', output_format='show'):

    plot_array(array=array, xticks=xticks, yticks=yticks, units=units, xyticksize=xyticksize,
               norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
               figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
               title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
               output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_noise_map_array(array, units, xticks, yticks, xyticksize=40,
                         norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                         title='Noise Map (electrons per second)', titlesize=46, xlabelsize=36, ylabelsize=36,
                         output_path=None, output_filename='noise_map', output_format='show'):

    plot_array(array=array, xticks=xticks, yticks=yticks, units=units, xyticksize=xyticksize,
               norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
               figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
               title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
               output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_psf_array(array, units, xticks, yticks, xyticksize=40,
                   norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                   title='PSF', titlesize=46, xlabelsize=36, ylabelsize=36,
                   output_path=None, output_filename='psf', output_format='show'):

    plot_array(array=array, xticks=xticks, yticks=yticks, units=units, xyticksize=xyticksize,
               norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
               figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
               title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
               output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_model_image_array(array, units, xticks, yticks, xyticksize=40,
                           norm='symmetric_log', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                           title='Model Image', titlesize=46, xlabelsize=36, ylabelsize=36,
                           output_path=None, output_filename='model_image', output_format='show'):

    plot_array(array=array, xticks=xticks, yticks=yticks, units=units, xyticksize=xyticksize,
               norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
               figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
               title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
               output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_residuals_array(array, units, xticks, yticks, xyticksize=40,
                         norm='symmetric_log', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
                         figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                         title='Residuals', titlesize=46, xlabelsize=36, ylabelsize=36,
                         output_path=None, output_filename='residuals', output_format='show'):

    plot_array(array=array, xticks=xticks, yticks=yticks, units=units, xyticksize=xyticksize,
               norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
               figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
               title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
               output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_chi_squareds_array(array, units, xticks, yticks, xyticksize=40,
                            norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
                            figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                            title='Chi Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
                            output_path=None, output_filename='chi_squareds', output_type='show'):

    plot_array(array=array, xticks=xticks, yticks=yticks, units=units, xyticksize=xyticksize,
               norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
               figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
               title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
               output_path=output_path, output_filename=output_filename, output_format=output_type)

def plot_scaled_noise_map_array(array, units, xticks, yticks, xyticksize=40,
                                norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                                figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                title='Scaled Noise Map (electrons per second)', titlesize=46, xlabelsize=36, ylabelsize=36,
                                output_path=None, output_filename='scaled_noise_map', output_format='show'):

    plot_array(array=array, xticks=xticks, yticks=yticks, units=units, xyticksize=xyticksize,
               norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
               figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
               title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
               output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_scaled_chi_squareds_array(array, units, xticks, yticks, xyticksize=40,
                                   norm='linear', norm_min=None, norm_max=None, linthresh=0.001, linscale=0.001,
                                   figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                   title='Scaled Chi Squareds', titlesize=46, xlabelsize=36, ylabelsize=36,
                                   output_path=None, output_filename='scaled_chi_squareds', output_type='show'):

    plot_array(array=array, xticks=xticks, yticks=yticks, units=units, xyticksize=xyticksize,
               norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
               figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
               title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
               output_path=output_path, output_filename=output_filename, output_format=output_type)

def plot_array(array, units, xticks, yticks, xyticksize,
               norm, norm_min, norm_max, linthresh, linscale,
               figsize, aspect, cmap, cb_ticksize,
               title, titlesize, xlabelsize, ylabelsize,
               output_path, output_filename, output_format):

    xlabel, ylabel = util.get_xylabels(units)

    norm_min, norm_max = util.get_normalization_min_max(array=array, norm_min=norm_min, norm_max=norm_max)
    norm_scale = util.get_normalization_scale(norm=norm, norm_min=norm_min, norm_max=norm_max,
                                            linthresh=linthresh, linscale=linscale)

    util.plot_image(array, figsize=figsize, aspect=aspect, cmap=cmap, norm_scale=norm_scale)
    util.set_ticks(array=array, xticks=xticks, yticks=yticks, xyticksize=xyticksize)
    util.set_title_and_labels(title=title, xlabel=xlabel, ylabel=ylabel, titlesize=titlesize,
                              xlabelsize=xlabelsize, ylabelsize=ylabelsize)
    util.set_colorbar(cb_ticksize=cb_ticksize)
    util.output_array(array=array, output_path=output_path, output_filename=output_filename,
                      output_format=output_format)

    plt.close()