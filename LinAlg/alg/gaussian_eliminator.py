#!/usr/bin/python
# July 2023  JMA
# gaussian_elimination.py  Style file for python modules
'''
Write a short description of what the program does here. 

Usage:
$ ./template.py [-v] [-l config_dir] [-o output_directory] input_file.csv
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
This program runs a read-eval-print loop
where you specify the dimensions of a random matrix. It creates
the matrix, then computes its LU decomposition matrici.

The prompt accepts two integers, the number of rows and columns.
If you just give one integer it creates a symmetric matrix.'''

########################################################################
class Ctemplate (object):
    'Module names and class names can be the same.'

    def __init__(self):
        self.input = None

    def rd_matrix(self, mat_file):
        'Input a tab-delimited integer file'
        try:
            with open(mat_file, 'rb') as fd:
                mat = pd.read_csv(fd, delimiter='\t')
        except Exception as e:
            print >> sys.stderr, 'Failed to read ', e, mat_file
            sys.exit(-1)
        self.input = mat

    def process(self):
        'Null operation'
        self.output = self.input

    def wr_header(self, fd, args):
            fd.write('# ' + '\t'.join(sys.argv) + '\n')
            host = re.sub('\n', '', subprocess.check_output('hostname'))
            user = os.environ['USER']
            date = time.asctime()
            fd.write('# ' + host+ '\t'+ user+ '\t'+ date + '\n')
        
    def wr_matrix(self, output_fn):
        try:
            out_fd = open(output_fn, 'wb')
            self.wr_header(out_fd, sys.argv)
            np.savetxt(out_fd, self.output, delimiter='\t', fmt='%8d')
        except Exception as e:
            print >> sys.stderr, 'Failed to write ', e, output_fn
            # out_fd.close()
            sys.exit(-1)
        out_fd.close()


########################################################################
def is_full_rank(the_m, d):
    return np.linalg.matrix_rank(the_m) == d

def random_ar(r,c):
    ar = np.reshape( np.array(np.random.choice(range(AR_MIN, AR_MAX), r*c, replace=True)), (r, c))
    # Check if not full rank
    return ar

def random_symmetric_ar(d):
    diagonal = np.random.choice(range(AR_MIN, AR_MAX), d, replace=True) 
    off_diagonal = list(np.random.choice(range(AR_MIN, AR_MAX), int(0.5 * d * (d-1)), replace=True) - round(d/2) )   # Yes the count is always an integer
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
    return ar

########################################################################
def set_limits():
    'Query the user for random digit limits'
    print(Fore.CYAN)
    limit = input('min, max? ')
    limits = re.findall(r'(\d+)', limit)
    AR_MAX = int(limits[1])
    AR_MIN = int(limits[0])
    print(Fore.RESET)

########################################################################
def command_loop(args):
    end_session = False
    while not end_session:
        print(Fore.RESET)
        response = input('\n    ')
        # parse input. 
        if len(response) == 0: 
            # end_session = True
            break
        elif len(response)  < 2:
            ar = random_symmetric_ar(int(response))
            print(Fore.CYAN)
            print(ar)
        else:
            responses = re.findall(r'(\d+)', response)
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


###############################################################################
def main(args):

    # Gassian elimination on a singular matrix
    A = np.array([[3,2,4],[1,2,2], [1,0,1]])
    b = np.transpose(np.array([[0,0,0]]))
    ## Run 

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
