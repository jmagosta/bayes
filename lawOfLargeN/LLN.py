#!/usr/bin/env python
# coding: utf-8

# # A visualization of the Law of Large Numbers & Central Limit Theorum
# To run remotely (Use your local IP as the origin address):
#
#  > bokeh serve LLN.py --allow-websocket-origin=192.168.15.100:5006
# 
# The remote client runs this in the browser as
# http://192.168.15.100:5006
# 
# To run locally just invoke
#
# > bokeh serve LLN.py
#
# To kill the process in powershell try cntl-del

import os, re, sys, time
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp

from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, MultiLine, PreText, Slider # Grid, LinearAxis, 

description = '''
The "law of large numbers" reveals that averages of sample items (e.g. individuals)
drawn from a population converge to a common value, the population mean, as 
the size of the sample increases. Increase the number of individuals per sample to 
see this. Increase the number of sample draws to get a better estimate of the 
distribution of the so-called sample statistic, without changing the distribution.
Note what happens when both sliders are set to 1.
'''

Population_Range = [0, 100]
# Slider parameters
sample_size = 12
number_of_samples = 1000
current_sample_ar = np.zeros((Population_Range[1] - Population_Range[0],), dtype='int')
samples_ar = np.zeros((Population_Range[1] - Population_Range[0],), dtype='int')

def create_a_sample(current_sample, pop, s_size):
    # Assume everything is global
    s = np.random.uniform(low=pop[0], high=pop[1], size=s_size)
    s_int = [int(z) for z in s]
    # Change it to a vector of counts
    for k in s_int:
        current_sample[k] +=1
    return np.mean(s), current_sample

def create_pop_samples(n_of_samples, pop, s_size):
    current_sample = np.zeros((Population_Range[1] - Population_Range[0],), dtype='int')
    samples = np.zeros((Population_Range[1] - Population_Range[0],), dtype='int')
    for _ in range(n_of_samples):
        sample_mean, most_current = create_a_sample(current_sample, pop, s_size)
        samples[int(sample_mean)] +=1
    return samples, most_current

def recompute_histogram(samples, cds, pop=Population_Range):
    'Update the plot values in the Column Data Source'
    xpts = [[k,k] for k in range(pop[0], pop[1])]
    ypts = [[0,0]] * (pop[1]- pop[0])
    sample_norm = samples.mean()
    print(f'recompute {sample_norm}')
    for i, a_sample in enumerate(samples):
        ypts[i]  = [0, a_sample/sample_norm]
    # MOdify the existing ColumnDataSource
    cds.data = dict(xs=xpts, ys=ypts)
    return cds

def plot_rug(src, subtitle= 'Histogram', plot_height = 160, line_width=3, line_color='grey'):
    'Create a histogram figure'
    max_y = max([z[1] for z in src.data["ys"]])
    # print(f'plot_rug {max_y}')
    p = figure( title=subtitle,
            width = 800, height = plot_height, 
            background_fill_color="#fafafa",
            x_range=Population_Range, 
            # y_range = (0, max_y),  # Unnecessary
            toolbar_location = None,
            )
    p.grid.grid_line_color="#fafafa"

    glyph = MultiLine(xs='xs', ys='ys', line_width=line_width, line_color=line_color)
    p.add_glyph(src, glyph )
    return(p)

def panel():
    'Assemble widgets and plots '
    global sample_size_slider, number_of_samples_slider 
    #print(f'panel means: {samples_ar.mean()} {current_sample_ar.mean()}')
    # kwargs = dict(subtitle= 'Population', plot_height = 160, line_width=3, line_color='grey')
    text_area = PreText(text=description, width=400)
    plots = column(
        plot_rug(pop_cds, subtitle="Distribution of Sample Averages", plot_height = 400, line_color='lightblue'),
        plot_rug(sample_cds, subtitle='Distribution of All Individuals Sampled.'))
    s_panel = row(plots, column(sample_size_slider, number_of_samples_slider, text_area, width=400))
    return s_panel


def update_sampling(attrname, old, new):
    '''callback
    attrname = the name of the quantity read from the widget
    old = its previous value
    new = the newly returned value
    '''
    global sample_size_slider, number_of_samples_slider, sample_cds, pop_cds 
    # Get current slider values (We could instead use the new arg value)
    # print(f'update_sampling {attrname}, {old}, {new}')
    ss = sample_size_slider.value
    ns = number_of_samples_slider.value
    # Recompute samples
    samples_ar, current_sample_ar = create_pop_samples( 
            ns, 
            Population_Range, 
            ss)
    sample_cds = recompute_histogram(current_sample_ar, sample_cds)
    pop_cds = recompute_histogram(samples_ar, pop_cds)

#########################################################
### Main  ###

#Initial values of the data
samples_ar, current_sample_ar = create_pop_samples( 
        number_of_samples, 
        Population_Range, 
        sample_size)

#Plot data structures
sample_cds = ColumnDataSource(dict(xs=[], ys=[]))
pop_cds = ColumnDataSource(dict(xs=[], ys=[]))

# Set plot values
sample_cds = recompute_histogram(current_sample_ar, sample_cds)
pop_cds = recompute_histogram(samples_ar, pop_cds)
# Set the callback

sample_size_slider = Slider(title="Number of Individuals per Sample", value=sample_size, start=1, end=200, step=1)
number_of_samples_slider = Slider(title="Number of Sample Draws", value=number_of_samples, start=1, end=2000,step=1)
# First arg is the trigger - the variable - value from the Widget
sample_size_slider.on_change('value', update_sampling)
number_of_samples_slider.on_change('value', update_sampling)
s_panel = panel()
curdoc().add_root(s_panel)
curdoc().title = "Law of Large Numbers Simulation"
