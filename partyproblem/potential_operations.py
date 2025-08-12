# Potential Operations on ID nodes
# used to solve the influence diagrams
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

def absorb_parent(the_parent, the_child):
    '''This removes conditioning by the parent.
    In the child potential, if necessary, move the parent dimension just
    before the marginal (typically the last dimension). This makes it
    possible to join the two potentials by multiplication, then to 
    absorb the parent by marginallizing out it's dimension. 
    '''
    if promote_conditional_dimension(the_parent.get_potential().get_named_dims(), 
                                     the_child.get_potential().get_named_dims()):
        # apply transpose to the parent dimensions
        pass
    # modify the child node TODO Does broadcasting work if the parent has parents? 
    combined_potential = join(the_parent.get_potential(), the_child.get_potential())
    # combined_potential = marginalize(combined_potential, -2)  # TODO Or pass it the name of the parent? 

    # TODO modify the child node?
    return combined_potential # ID_node()

def promote_conditional_dimension(parent_dims, child_dims):
    '''If the matching last dimension of the parent - assumed
    to be the marginal is not at the end of the list of conditionings
    for the child, find the transpose to move it.'''
    # The parent marginal is the last variable in the dimensions
    parent_marginal = list(parent_dims.keys())[-1]
    child_variables = list(child_dims.keys())
    # It must be one of the conditioning variables of the child:
    if not (parent_marginal in child_variables[:-1]):
        print(f'{parent_marginal} does not condition {child_variables}' )
    elif True:
    # TODO if the conditioning variable is not second to last, return 
    # the required transpose
        pass
    else:
    # its already at the end, so do nothing.
        return None

def join(parent_p, child_p): # parent, child
    '''Lower level function to multiply potentials once dimensions
    are aligned.'''
    cpt = (parent_p.cpt.unsqueeze(-1) * child_p.cpt).sum(-2)
    # remove the conditioning variable 
    # -- assumed to be the last (marginal) var of the parent. 
    named_sh = child_p.remove_dim(parent_p.index_named_dims(-1)[0])
    return Potential(cpt, named_sh)
    

def marginalize(the_potential, the_dimension):
    '''Lower level function to marginalize a potential by 
    removing a dimension.  Typically used after a join, resulting
     in the potential last dimension as a marginal.'''
    
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
