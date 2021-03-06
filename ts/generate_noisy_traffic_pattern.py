#!/usr/bin/env python20
# generate noisy traffic pattern
# JMA 1 Jan 20
# coding: utf-8

# # Noisy timeseries patterns
# A timeseries by minute, with trend, noise, and hourly and daily cycles

import re, os, sys
from pathlib import Path
import math
from datetime import datetime
import json
import numpy as np
import pandas as pd
import statsmodels.nonparametric.smoothers_lowess
# Check if its the right version
# statsmodels.__version__

from bokeh.plotting import figure, save, show, output_file
index = 0
MIN_PER_DAY = 24*60

###############################################################################
# Use np to broadcast a factor over lists * tuples
def m(factor, iterable):
    return tuple(factor * np.array(iterable))

class component(object):

    def __init__(self, ts, phase= 3*math.pi/2):
        self.phase = phase
        self.ts = ts

    def periodic_component(self, x, amplitude, period, phase):
        #print('amplitude, period', amplitude, period)
        #print('x, phase', x, phase)
        'Sine component of period cycles per day between 0 and amplitude, shifted by phase'
        component = 0.5 *amplitude *(1 + np.sin( 2 * math.pi * period * (x/MIN_PER_DAY) +  self.phase ))
        #print(component)
        return component

    #  closure
    def a_component(self, amp, freq):
        #print('amp, freq ', amp, freq)
        return self.periodic_component(self.ts, amp, freq, self.phase)


# Create an example 
def tst_example():
    'Return a dict of parameters.'
    parameters = dict()
    # Time base
    parameters["n_days"] = 1 # how many days do you want to run this pattern?
    # Amplitude
    parameters['max_amplitude'] = 1000  # 10 VMs 
    # Periodic components
    # daily_freq = 24 * 60
    daily_phase = math.pi
    daily_amplitude = 0.6 * parameters['max_amplitude']
    parameters['daily_components'] = (1,3, 4, 6) 
    parameters['daily_amplitudes'] = m(daily_amplitude, [1, 0.09, 0.1, 0.1])

    hourly_freq = 7.0  #24 == once an hour 
    hourly_amplitude = 0.08 * parameters['max_amplitude']
    hourly_phase = 0.0
    parameters['hourly_components'] = m(hourly_freq, [1,3])
    parameters['hourly_amplitudes'] = m(hourly_amplitude, [1, 0.19])
    # Trend
    parameters['rate'] = 0.3  # fraction of increase (decrease) per day
    parameters['offset'] = 20  # min value is 10 VMs at 2 %
    # Error term
    parameters['noise_lvl'] = 40.0
    return parameters

class noisyTraffic(object):

    def __init__(self, pms, index=0):
        ' pms - parameter dict, index - output file number.'
        # Output file
        index += index
        self.save_csv_file = f"P{index}_n{pms['n_days']}_e{pms['noise_lvl']}_d{pms['daily_amplitudes'][0]}_h{pms['hourly_amplitudes'][0]}_pattern.csv"

    def assembleComponents(self, pms):
        '''Assemble various cyclic, trend and noise components.
        pms - parameter dictionary''' 
        # Conversions
        duration = MIN_PER_DAY * pms['n_days'] # duration of the pattern in minutes
        ts = np.arange(duration)  # Time axis, integer minutes
        c = component(ts)

        # Trend
        series = np.array(pms['offset'] + pms['max_amplitude']* ts * pms['rate']/MIN_PER_DAY)
        # Daily components
        series +=  np.array(list(map(sum, zip(*[c.a_component(*z) for z in zip(pms['daily_amplitudes'], pms['daily_components'])]))))
        # Hourly components
        series +=  np.array(list(map(sum, zip(*[c.a_component(*z) for z in zip(pms['hourly_amplitudes'], pms['hourly_components'])]))))
        # Noise 
        rs = np.random.RandomState()
        normal_rnd = rs.standard_normal(duration)
        noisy_series = abs(series + pms['noise_lvl'] * normal_rnd)
        # Create numpy array
        self.pattern = np.stack([ts, noisy_series], axis=1)
        self.true_signal = np.stack([ts, series], axis=1)

        # Compare to a smoothed version 
        # estxy = statsmodels.nonparametric.smoothers_lowess.lowess(series, ts, frac=0.05, it=3, is_sorted=True)
        # sm_x, sm_y = list(zip(*estxy))
        # self.smoothed = np.vstack([np.array(sm_x), np.array(sm_y)])  # TODO find a cleaner way. 

def plt_fit(actual_ts, true_ts):
    # bokeh is working. 
    TOOLS = "hover,pan,box_zoom,save"
    # colors = ["#%02x%02x%02x" % (int(127 + 127*r),100, int(127 + 127*r)) for r in special.erf(normal_rnd)]
    colors = "#%02x%02x%02x"
    p1 = figure(width=800, height=300, tools=TOOLS)
    p1.scatter(actual_ts[:,0], actual_ts[:,1], color='grey', size=1.0,alpha=0.4)
    p1.line(true_ts[:,0], true_ts[:,1], line_color = 'Black')
    save(p1)

def wr_ts(actual_ts, true_ts, save_csv_file):
    # Create a two column csv file with header minutes = 0..1399, cpuload > 0 from a np array.
    # Creating pandas dataframe from numpy array
    # Assume actual_ts and true_ts have the same time column
    dataset = pd.DataFrame({'minutes': actual_ts[:,0], 'cpuload': actual_ts[:,1], 'trueload': true_ts[:,1]})
    dataset.to_csv(save_csv_file, index=False)
    print('# Wrote to: ', save_csv_file)

### MAIN
################################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1: # input input parameters from json
        init_start = json.load(sys.argv[1])

    # main both plots and writes out results
    p = tst_example()
    pattern = noisyTraffic(p)
    pattern.assembleComponents(p)
    #print(pattern.pattern)
    plot_filename = re.sub('.py$', '_graph.html', Path(__file__).name)
    output_file(filename= plot_filename, title='Plot')
    plt_fit(pattern.pattern, pattern.true_signal)
    print(f'Plot saved to {plot_filename}')
    wr_ts(pattern.pattern, pattern.true_signal, pattern.save_csv_file)

