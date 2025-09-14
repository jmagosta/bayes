# ID_operations.py
# JMA 16 Jul 2025
# from party_problem_xdsl_bis.ipynb

# TODO - move potential algebra to potential_operations

import torch

import networkx as nx

# Use to parse xdsl files
# import xml.etree.ElementTree as et
# for extract_net
from Potential import *
import BN

from tabulate import tabulate

###########

    
def get_potential(a_node, n_dict):
    'Find the probability np array in the node, and label it using parents in the graph'
    # The states of the RV label the columns, so that the matrix is row-markov
    the_cpt = n_dict[a_node]['potential']
    return the_cpt

# Format one-dim tensors 
# from collections import deque
def pr_one_dim_table(the_potential, the_var, n_dict, **args):
    def que_copy(prefix, queue):
       if not isinstance(queue, list):
           queue = [queue]
       queue.insert(0, prefix)
       return queue
    states = n_dict[the_var]['states']
    values = the_potential.tolist()
    # Flatten nested lists
    while len(values)  == 1:   # TODO is this test necessary?
        values = values[0]
    print(f' *** {the_var} ***')
    values = [que_copy(s, v) for s, v in zip(states, values)]
    print(tabulate(values, **args))

    ## For joining by aligning potentials as named tensors

def dim_index(potential_cpt, candidate):
    '''Find where a labeled candidate dimension is in the dimensions 
    of a potential, if it is not already last.'''
    # TODO how does this work if more than one dimension is found?
    # Starting with 0 as the first location, so the last location 
    # equals the length -1 of the shape
    cpt_dims = potential_cpt.dim_names
    # if its included is it not already last?
    if candidate in cpt_dims and candidate != list(cpt_dims)[-1]:
        return list(cpt_dims).index(candidate) 
    else:
        # Either candidate not found or its already at the end, so do nothing.
        return None

def move_named_dim_to_end(the_named_tensor, the_dimension):
    'Transpose the potential place the dimension last'
    the_dim_index = dim_index(the_named_tensor, the_dimension)
    if the_dim_index is not None:
        # Create a modified shape OrderedDict
        shape = the_named_tensor.shape.copy()
        shape.move_to_end(the_dimension)
        # Rotate the tensor dimensions
        p_transpose = list(range(len(shape)))               # The unperturbed list
        p_transpose.append(p_transpose.pop(the_dim_index))  # Move index to end
        # Transpose CPT
        x = the_named_tensor.p.permute(p_transpose) #TODO .p -> .cpt
        return Potential(x, shape)
    else:
        # A no op
        return the_named_tensor.p 
    
# No problem with mapping single arg functions over tensors!  
def delta_utility(x, exponand = 0.5, normalize = 50):
    dims = x.get_named_dims()    #x.shape
    u = 4/3*(1 - pow(exponand, (x.cpt/normalize)))
    return Potential(u, dims)

# TODO not used. 
def marginalize(child_potential, parent_potential):
    cpt = (child_potential.p * parent_potential.p).sum(-1)
    # TODO remove parent shape from child
    sh = OrderedDict(set(child_potential.shape.items()) - set(parent_potential.shape.items()))
    return Potential(cpt, sh)

def marginalize_last(p1, p2):
    '''For a potential matching the last dimension of the other, join them,
    then marginalized out the last dimension'''
    if list(p1.shape)[-1] != list(p2.shape)[-1]:           # Compare shapes by indexed value
        print(f'Err, last shapes do not match:\t{list(p1.shape)[-1]} != {list(p2.shape)[-1]}')
        return None
    else:
        # NOTE: pytorch does not restrict marginalization to just the last dim
        new_tensor = (p1.p * p2.p).sum(-1)
        # The symmetric set difference - those not common to both. 
        s1 = set(p1.shape.items())
        s2 = set(p2.shape.items())
        new_shape = OrderedDict(s1.union(s2) - s1.intersection(s2))
    return Potential(new_tensor, new_shape)

def shift_to_end(the_shape, the_var):
    the_shape.move_to_end(the_var)  # method from OrderedDict
    return the_shape


def join_parent(the_conditional, the_parent):
    'Assume the parent rv is the last dim in the conditional, and marginalize out that dim'
    # Find the parent and transpose it to last dim
    # c_potential = get_potential(the_conditional, name_dict)
    # p_potential = get_potential(the_parent, name_dict)
    # TODO make this work for more than one
    parent_var = list(the_parent.get_dim_names())[0]
    found_dim = dim_index(the_conditional, parent_var)
    # Is found dim not already in the last dim? 
    new_shape = the_parent.shape
    if found_dim is not None:   # TODO does this work if the found dim is first?
        # Move found_dim to last dimension
        new_shape = shift_to_end(new_shape, parent_var)
        c_transpose = list(range(len(new_shape)))
        c_transpose.append(c_transpose.pop(found_dim))
        # Transpose CPT
        the_conditional.p.permute(c_transpose)
        # TODO - create a new potential? 
    new_joint =  Potential(the_conditional.p * the_parent.p, new_shape)
    return new_joint

if __name__ == '__main__':

    a_potential = new_Potential([0.1, 0.9, 0.4, 0.6], [2,2], ['margin', 'condition'])
    p_potential = new_Potential([0.04, 0.96], [2,1], ['condition']  )
    print(p_potential.p * a_potential.p)
    print(marginalize_last(a_potential, p_potential))

  