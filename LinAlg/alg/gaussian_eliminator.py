#!/usr/local/bin/python3
# July 2023  JMA
# gaussian_eliminator.py 
'''
An interactive demonstration of Gaussian elimination

Usage:
$ ./gaussian_eliminator.py
''' 


### Python standard modules - remove those not necessary
import copy
import math
import os
import subprocess
import sys
import re
import string
import time
import unittest
from colorama import Fore, Back, Style

import numpy as np
# A case where the top level package needs it's subpackages imported explicitly. 
import scipy.linalg    
import pandas as pd

### config constants 
VERBOSE = False
AR_MIN = 0
AR_MAX = 4

INTRO_PROMPT = '''
Welcome to the Gaussian elimination solver! 

This program runs a read-eval-print loop
where you specify the dimensions of a random matrix. It creates
the matrix, then computes its LU decomposition matrici.

First it asks you for the minimum and maximum value for the random
matrix elements.  Then you enter the loop where it prompts you 
for the matrix dimensions. The prompt accepts two integers, 
the number of rows and columns. If you just give one integer 
it creates a symmetric matrix. Just hit return to end the program. 

Also, if the random matrix turns out to not be full rank, it will tell you.
'''

########################################################################
def is_full_rank(the_m, d):
    return np.linalg.matrix_rank(the_m) == d

def random_ar(r,c):
    ar = np.reshape( np.array(np.random.choice(range(AR_MIN, AR_MAX+1), r*c, replace=True)), (r, c))
    # Check if not full rank
    if not is_full_rank(ar, min(r,c)):
        print(f'This matrix {r},{c} is not full rank.')
    return ar

def random_symmetric_ar(d):
    diagonal = np.random.choice(range(AR_MIN, AR_MAX+1), d, replace=True) 
    off_diagonal = list(np.random.choice(range(AR_MIN, AR_MAX+1), int(0.5 * d * (d-1)), replace=True))   # Yes the count is always an integer
    # Clever way to fill the matrix?  Iterate thru the off diagonal lower matrix and pop the off diagnozal elements
    ar = np.zeros((d,d))
    # Set the lower off diagonal, then use the transpose to create the upper
    for a_row in range(1,d):
        for a_col in range(a_row):
            ar[a_row, a_col] = off_diagonal.pop()
    ar = ar + ar.T
    # Set the diagonal
    for k in range(d):
        ar[k,k]  = diagonal[k]
    # Check if not full rank
    if not is_full_rank(ar, d):
        print('This symmetric matrix is not full rank.')
    return ar

########################################################################
def set_limits():
    'Query the user for random digit limits'
    global AR_MAX, AR_MIN
    print(Fore.CYAN)
    limit = input('min, max? ')
    limits = re.findall(r'(-?\d+)', limit)
    AR_MAX = int(limits[1])
    AR_MIN = int(limits[0])
    print(Fore.RESET)

########################################################################
def command_loop(args):
    end_session = False
    while not end_session:
        print(Fore.RESET)
        response = input('\nDimensions? ')
        # parse input. 
        if len(response) == 0: 
            print("Goodbye!\n")
            break
        responses = [int(z) for z in re.findall(r'(\d+)', response)]
        if len(responses)  < 2:
            ar = random_symmetric_ar(responses[0])
            print(Fore.CYAN)
            print(ar)
        else:
            ar = random_ar(int(responses[0]), int(responses[1]))
            print(Fore.CYAN)
            print(ar)
        skip = input(Fore.CYAN + "\n<cr> to solve. ")
        if not skip:
            # The LU decomposition - Gaussian elimination
            P, L, U = scipy.linalg.lu(ar)
            # View the Gassian elimination results for matrix A
            print(Fore.LIGHTBLACK_EX + '\nP =')
            print(P)
            print(Fore.GREEN + '\nL =')
            print(np.round(L,2))
            print(Fore.RED + '\nU =')
            print(np.round(U,2))



########################################################################
if __name__ == '__main__':

    if '-v' in sys.argv:
        k = sys.argv.index('-v')
        VERBOSE = True
        del(sys.argv[k:(k+2)])

    args = sys.argv[1:]
    print(INTRO_PROMPT)
    st = time.time()
    set_limits()
    z = command_loop(args)
    print(Fore.RESET + sys.argv[0], f"\tDone in {time.time() - st:5.1} secs!", file=sys.stderr)
    sys.exit(z)

#EOF
