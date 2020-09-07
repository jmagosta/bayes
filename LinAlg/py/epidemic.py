# epidemic.py
# Sept 2020  (c) John M Agosta
# Compute the trajectory of a SIR epidemic model 
'''
This program illustrates computation of a well-known non-linear differential equation
called the SIR (Suceptible-Infected-Recovered) model of epidemic infections. It solves
this set of equations - where ' indicates the time derivative:

    ds/dt = -Rsi
    di/dt =  Rsi - Qi
    dr/dt =  Qi

where R is the infection parameter, the rate at which combinations of suceptables s
and infectives i result in new infections, and Q is the removal parameter rate, at which
infectives, i, become immune. 

To run this program:

    python epidemic.py R Q [n_days] [time_increment] [output_file]

Assume an initial population of 1000
'''
import math, os, sys
from pathlib import Path


# Some default parameters
VERBOSE = False
population = 1
infection_fraction = 0.001
n_days = 100             # The number of days to run the simulation.
time_increment = 0.5     # Update values every this fraction of a day.
R = 1.1
Q = 2.0
n_print_cols = 120       # Scale the output to fit in this number of columns

### simulation functions
def increment_s(s, i, R):
    s = s - R * s * i
    return s

def increment_i(s, i, r, Q):
    i = i + R * s * i - Q * i
    return i

def increment_r(i, r, Q):
    r = r + Q * i
    return r

def run_simulation(s,i,r, R, Q):
    for k in range(N-1):
        s[k+1] = increment_s(s[k], i[k], R)
        i[k+1] = increment_i(s[k], i[k], R, Q)
        r[k+1] = increment_r(i[k], r[k], Q)
    return s, i, r

### display functions  

def sort_vectors(s, i, r):
    'At each instant, order the vectors by value'
    # Do this with the 3 pair-wise comparisons?
    s_less_i, s_less_r, i_less_r = False, False, False
    if s < i:
        s_less_i = True
    if s < r:
        s_less_r = True
    if i < r:
        i_less_r = True
    # And six possible outcomes
    if not s_less_r and s_less_i:
        return ['r', 's', 'i']
    elif s_less_r and not s_less_i:
        return ['i', 's', 'r']
    elif s_less_i and i_less_r:
        return ['s', 'i', 'r']
    elif not s_less_i and not i_less_r:
        return ['r', 'i', 's']
    elif  s_less_r and not i_less_r:
        return ['s', 'r', 'i'] 
    elif  not s_less_r and i_less_r:
        return ['i', 'r', 's'] 
    else:
        print ('Oops! Sorting failed')

def second_min(the_set, min_value):
    'The second smallest value in the set'
    the_set.remove(min_value)
    return min(*the_set)

def pr_plot(s, i, r):
    'Draw an ascii graph of s,i,r with time on the vertical axis'
    for k in range(N-1):
        if VERBOSE:
            print(f's {float(s[k]):.2}, i {float(i[k]):.2}, r {float(r[k]):.2}:', end='')
        svs = sort_vectors(s[k], i[k], r[k])
        first_col = min(s[k], i[k], r[k])
        print(round(n_print_cols * first_col) * '-', end='')
        print(svs[0], end='')
        second_col = second_min([s[k], i[k], r[k]], first_col)
        print(round(n_print_cols * second_col) * '.', end='')
        print(svs[1], end='')
        print(round(n_print_cols * (max(s[k], i[k], r[k]))) * ' ', end='')
        print(svs[2])

if __name__ == '__main__':

    number_of_arguments = len(sys.argv) - 1
    # No command line arguments, just print the documentation
    if number_of_arguments < 2:
        print(__doc__)
    if number_of_arguments >= 2:
        R = float(sys.argv[1])
        Q = float(sys.argv[2])
    if number_of_arguments >= 3:
        n_days = int(sys.argv[3])
    if number_of_arguments >= 4:
        time_increment = float(sys.argv[4])
    if number_of_arguments == 5:
        output_file = Path(sys.argv[5])

    # Output vectors
    N = math.floor(n_days/time_increment)
    s = N * [0]
    i = N * [0]
    r = N * [0]
    s[0] = population
    i[0] = infection_fraction

    s, i, r = run_simulation(s, i, r, R, Q)

    pr_plot(s, i, r)