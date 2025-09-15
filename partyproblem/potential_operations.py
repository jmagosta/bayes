# Potential Operations for solving BN variable reduction
# used to solve the influence diagrams, as the lower
# level to node operations. 
#  JMA 9 Aug

# def dim_index(potential_cpt, candidate):

# def move_named_dim_to_end(the_named_tensor, the_dimension):

# def marginalize_last(p1, p2):

# def marginalize(child_potential, parent_potential):

# def shift_to_end(the_shape, the_var):

# def join_parent(the_conditional, the_parent, name_dict= bn.n_dict):

### named dimensions operations, for OrderedDict objects (TODO - dicts might also work)

# from Potential import *

### Bayes net solution operations

import numpy as np

from Potential import *

def absorb_parent(the_parent_p: Potential, the_child_p:Potential):
    '''This removes conditioning by the parent
    in the child potential. If necessary, move the parent dimension just
    before the marginal (by convention the last dimension). This makes it
    possible to join the two potentials by multiplication, then to 
    absorb the parent by marginalizing out it's dimension. 
    '''
    updated_child_permutation, conditioning_var = promote_conditional_dimension(the_parent_p.get_named_dims(), 
                                     the_child_p.get_named_dims())
    if updated_child_permutation is None:
        # Its  not a parent of child, do nothing
        pass
    elif (updated_child_permutation != list(range(len(the_child_p.get_named_dims())))):
        # If its not an identity permutation, apply it.
        the_child_p = the_child_p.transpose_named_tensor(updated_child_permutation) 
    # TODO Does broadcasting work if the parent has parents? 
    combined_potential = join(the_parent_p, the_child_p)
    # combined_cpt = marginalize(combined_potential, -2)  # TODO Or pass it the name of the parent? 
    # The child node inherits conditionings and preserves its other conditionings
    # and the parent node can be discarded. 
    reduced_potential = marginalize(combined_potential, conditioning_var)
    # TODO  return a modified child Potential 
    # return Potential(cpt, updated_named_dims)
    return reduced_potential 

def swap_indexes(i_1, i_2, indexes):
    indexes[i_1], indexes[i_2] = indexes[i_2], indexes[i_1]
    return indexes

def promote_conditional_dimension(parent_dims, child_dims):
    '''If the matching last dimension of the parent - assumed
    to be the marginal is not at the end of the list of conditionings
    for the child, find the transpose to move it.
    Return the updated child conditional dimensions.'''
    # Dont assume the parent marginal is the last variable in the dimensions
    child_conditionals = [m_var[0] for m_var in child_dims.items() if m_var[1] == 'c']
    parent_marginal = [c_var[0] for c_var in parent_dims.items() if c_var[1] in ('m', 'v')][0]
    # It must be one of the conditioning variables of the child:
    if not (parent_marginal in child_conditionals):
        print(f'{parent_marginal} not contained in {child_conditionals} conditionals' )
        # Not a parent, make no change to the child
        return None, None
    else:
        # compute the permutation for the child conditionals
        cc_permutation = list(range(len(child_dims)))
        conditioning_var = child_conditionals[-1]
        if conditioning_var != parent_marginal:
            i_child_conditionals = child.find_var(parent_marginal)
            # move it to the end of the child named_dims
            cc_permutation = swap_indexes(i_child_conditionals, 
                                          len(cc_permutation)-2, # place the matching conditional second to last
                                          cc_permutation) 
            # update the conditioning var
            conditioning_var = parent_marginal              
        return cc_permutation, conditioning_var
    
def drop_singleton_dimension(the_potential):
    'Used to reduce utility for taking expectations'
    # As an alternative, don't add the singleton in the first place
    # Find the singleton dim, assuming only one singleton index
    singleton_id = the_potential.get_dim_sizes().index(1)
    # Get the name of that singleton dim
    singleton_name = the_potential.get_dim_names()[singleton_id]
    # Remove it from the dims
    reduced_dims = the_potential.get_named_dims().copy()
    del reduced_dims[singleton_name]
    # x = the_potential.cpt.squeeze(1)
    return Potential(the_potential.cpt.squeeze(singleton_id), reduced_dims)


def join(parent_p, child_p): # parent, child
    '''Lower level function to multiply potentials once dimensions
    are aligned. Creates a joint of the parent and child.'''
    cpt = (parent_p.cpt.unsqueeze(-1) * child_p.cpt)
    # TODO extend to a parent with conditionings. 
    # For now assume only the child has conditionings
    child_named_dim = child_p.get_named_dims().copy()
    # The conditioning var is now also a marginal, assume its second to last
    child_named_dim[parent_p.get_dim_names()[-1]] = 'm'
    return Potential(cpt, child_named_dim)
    
def marginalize(the_potential, absorbed_var):
    '''Lower level function to marginalize a potential by 
    removing a dimension.  Typically used after a join. Marginalize
    out the parent to remove it, or the child, to reverse the arc.'''
    # remove the conditioning variable 
    # -- assumed to be the last (marginal) var of the parent.
    absorbed_dim = the_potential.find_var(absorbed_var)
    new_potential = the_potential.cpt.sum(absorbed_dim)
    named_sh = the_potential.remove_dim(absorbed_var)
    return Potential(new_potential, named_sh) 

# No problem with mapping single arg functions over tensors! 
def named_tensor_apply(a_potential, transformation, **kwargs): 
    dims = a_potential.get_named_dims()
    # Apply the utility function
    a_tensor = a_potential.cpt 
    u = transformation(a_potential.cpt, **kwargs)
    # u = 4/3*(1 - pow(kwargs['exponand'], (a_potential.cpt/kwargs['normalize'])))
    return Potential(u, dims)

def delta_utility(a_value, **kwargs):
    'Tranformation from value to utility'
    # dims = a_potential.get_named_dims()
    a_utility = 4/3*(1 - pow(kwargs['exponand'], (a_value/kwargs['normalize'])))
    return a_utility 

def delta_inverse_utility(a_utility, **kwargs):
    'Inverse transformation from utility back to value'
    a_value = kwargs['normalize'] * np.log(1 - 3*a_utility/4)/np.log(kwargs['exponand'])
    return a_value

### Main

if __name__ == '__main__':

    # from Potential import * 

    utils = [10, 9, 1]
    u_potential = new_Potential(utils, [3,1], ['uncertainty', 'value'])
    u_potential.pr_potential()

    z = named_tensor_apply(u_potential, delta_utility, exponand = 0.5, normalize = 50)
    print(z)
    drop_singleton_dimension(u_potential).pr_potential()
    print('\n')

    probs = [ r for p in [0.9,  0.1, 0.0,  1.0, 0.3, 0.7] for r in (p, 1-p)]
    # Place margin probabilities in the last dimension
    child = new_Potential(probs, 
                       [2,3,2], 
                       ['predictor', 'condition1', 'margin'])
   


    parent = new_Potential([1.0, 0.0],
                            [2],
                            ['predictor'])

    parent.pr_potential()
    print('\n')
    child.pr_potential()
    print('\n')
    # p = promote_conditional_dimension(parent_node.get_potential().get_named_dims(), 
    #                                  child_node.get_potential().get_named_dims())
    absorb_parent(parent, child).pr_potential()
