# Potential Operations for solving BN variable reduction
# used to solve the influence diagrams, as the lower
# level to node operations. 
#  JMA 9 Aug

# def dim_index(potential_cpt, candidate):

# def move_named_dim_to_end(the_named_tensor, the_dimension):

# def marginalize_last(p1, p2):

# def delta_utility(x, exponand = 0.5, normalize = 50):

# def marginalize(child_potential, parent_potential):

# def shift_to_end(the_shape, the_var):

# def join_parent(the_conditional, the_parent, name_dict= bn.n_dict):

### named dimensions operations, for OrderedDict objects (TODO - dicts might also work)

# from Potential import *

### Bayes net solution operations

def absorb_parent(the_parent_p, the_child_p):
    '''This removes conditioning by the parent
    in the child potential. If necessary, move the parent dimension just
    before the marginal (by convention the last dimension). This makes it
    possible to join the two potentials by multiplication, then to 
    absorb the parent by marginalizing out it's dimension. 
    '''
    updated_child_permutation = promote_conditional_dimension(the_parent_p.get_named_dims(), 
                                     the_child_p.get_named_dims())
    if updated_child_permutation is None:
        # Its  not a parent of child, do nothing
        pass
    elif (updated_child_permutation != list(range(len(the_child_p.get_named_dims())))):
        # If its not an identity permutation, apply it.
        the_child_p = self.the_child_p.transpose_named_tensor(updated_child_permutation) 
    # TODO Does broadcasting work if the parent has parents? 
    combined_cpt = join(the_parent_p, the_child_p)
    # combined_cpt = marginalize(combined_potential, -2)  # TODO Or pass it the name of the parent? 
    # The child node inherits conditionings and preserves its other conditionings
    # and the parent node can be discarded. 
    updated_named_dims = merge_conditionings()
    # TODO  return a modified child Potential 
    # return Potential(cpt, updated_named_dims)
    return combined_potential 

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
    parent_marginal = [c_var[0] for c_var in parent_dims.items() if c_var[1] == 'm'][0]
    # It must be one of the conditioning variables of the child:
    if not (parent_marginal in child_conditionals):
        print(f'{parent_marginal} not contained in {child_conditionals} conditionals' )
        # Not a parent, make no change to the child
        return None
    else:
        # compute the permutation for the child conditionals
        cc_permutation = list(range(len(child_dims) -1))
        if child_conditionals[-1] != parent_marginal:
            i_child_conditionals = child_dims.find_var(parent_marginal)
            # move it to the end of the child named_dims
            cc_permutation = swap_indexes(i_child_conditionals, 
                                          len(cc_permutation)-1,
                                          cc_permutation)               
        return cc_permutation

def join(parent_p, child_p): # parent, child
    '''Lower level function to multiply potentials once dimensions
    are aligned. Creates a joint of the parent and child.'''
    cpt = (parent_p.cpt.unsqueeze(-1) * child_p.cpt)
    return cpt
    
def marginalize(the_potential, the_dimension, absorbed_dim):
    '''Lower level function to marginalize a potential by 
    removing a dimension.  Typically used after a join. Marginalize
    out the parent to remove it, or the child, to reverse the arc.'''
    # remove the conditioning variable 
    # -- assumed to be the last (marginal) var of the parent.
    new_potential = the_potential.sum(absorbed_dim)
    named_sh = child_p.remove_dim(parent_p.index_named_dims(-1)[0])  

### Main

if __name__ == '__main__':

    from Potential import *  # Includes ID_node module

    parent_features = dict(name='parent',
                           kind = 'cpt',
                           parents = [],
                           states = ['absent', 'present'],
                           potential = new_Potential([0.9, 0.1],
                                                     [2],
                                                     ['predictor']))

    parent_node = create_from_dict(parent_features)

    child_features = dict(name='parent',
                           kind = 'cpt',
                           parents = [],
                           states = ['false', 'true'],
                           potential = new_Potential([0.51, 0.49, 0.01, 0.99],
                                                     [2, 2],
                                                     ['predictor', 'outcome']))

    child_node = create_from_dict(child_features)

    parent_node.pr_node()
    child_node.pr_node()

    # p = promote_conditional_dimension(parent_node.get_potential().get_named_dims(), 
    #                                  child_node.get_potential().get_named_dims())
    print(absorb_parent(parent_node, child_node))
