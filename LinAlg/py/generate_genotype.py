# generate_genotype.py
# JMA 17 Aug 2020
# A random creation of a genotype encoding

import os, sys
import random

file_name = "genotype_Dad.txt"
hybrid_count = 10

header ='''
# This is a comment -- have your program ignore all lines that begin with a '#'
#
# The following lines contain an encoding of the parents genotypes.
# Each line represents one allele, made up of a pair of two traits, either of which
# may be dominant, represented by 'D' or recessive, by 'r'.  It the traits
# on a line are different, that allele is hybrid.
#
# The possible offspring of the parents consist randomly of one trait
# from each allele. If the allele is hybrid then there are two possible
# offspring phenotypes that are possible for each allele. 
#
# The computational task is to generate all possible offspring phenotypes
# from this genotype. Since there are many hybrid alleles in the sequence
# the number of possibilities will be large. Your program should generate
# all possible offspring combinations. For example, with this trivial example
# genotype:
#
# D D
# r D
# D r
#
# Results in these 4 phenotypes:
#
# D  D  D  D
# r  r  D  D
# r  D  r  D

'''

def one_allele(n):
    ''
    # hybrid or not
    if random.randint(0,1):
        the_allele = random.choice(['r D', 'D r'])
        n +=1 
    else:
        # 
        the_allele = random.choice(['r r', 'D D'])
    return the_allele, n

n = 0
output = open(file_name, 'wt')
print(header, file=output)

while n < hybrid_count:
    a, n = one_allele(n)
    print(a, file=output)