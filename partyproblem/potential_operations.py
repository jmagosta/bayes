# Potential Operations for solving BN variable reduction
# used to solve the influence diagrams, as the lower
# level to node operations. 
#  JMA 9 Aug

# def dim_index(potential_cpt, candidate):

# def move_named_dim_to_end(the_named_tensor, the_dimension):

# def marginalize_last(p1, p2):

# def marginalize(_out(child_potential, parent_potential):

# def shift_to_end(the_shape, the_var):

# def join_parent(the_conditional, the_parent, name_dict= bn.n_dict):

### named dimensions operations, for OrderedDict objects (TODO - dicts might also work)

# from Potential import *

### Bayes net solution operations

import numpy as np

from Potential import *

def reverse_parent(the_parent_p: Potential, the_child_p:Potential) -> list[Potential]:
    '''After creating the joint, return the parent as child and child as new parent'''
    #Check if either are decision nodes.  Few ops are allowed on decisions
    # Call join parent
    # Marginalize to create the new parent
    # Marginalize to create the new child
    return None

def absorb_parent(the_parent_p: Potential, the_child_p:Potential):
    '''This removes conditioning by the parent
    in the child potential. If necessary, move the parent dimension just
    before the marginal (by convention the last dimension). This makes it
    possible to join the two potentials by multiplication. Then, to 
    absorb the parent, marginal out it's dimension. 
    Result:
        The child Potential, with the parent var removed. 
    '''
    # Simple case: parent has no conditionings. 
    if the_parent_p.rank() == 1:
        if the_child_p.cpt.shape[-1] != 1:
            updated_child_permutation, conditioning_var =\
                promote_conditional_dimension(the_parent_p.get_named_dims(),\
                                                the_child_p.get_named_dims()) 
            if updated_child_permutation is None:
            # Its  not a parent of child, return the unmodified child 
            # (TODO: would this ever happen?)
                return the_child_p
            # Else if parent is not already the second to last dim
            elif updated_child_permutation != list(range(the_child_p.rank())):
                the_child_p = the_child_p.permute_named_tensor(updated_child_permutation)
            combined_potential = join(the_parent_p, the_child_p)
            reduced_potential = marginalize_out(combined_potential, conditioning_var)
        else: # Special case for utility nodes that have a unit last dimension
            reduced_potential = marginalize_utility( the_child_p, the_parent_p)

    else:  
        print(f'Absorb_parent(): Not implemented when parent {the_parent_p.get_named_dims()} has conditionings')
        reduced_potential = None
    # updated_child_permutation, conditioning_var =\
    #     align_dimensions(the_parent_p.get_named_dims(), 
    #                                  the_child_p.get_named_dims())
    # TODO The child node inherits the parent conditionings and preserves its other conditionings
    # The parent node can be discarded. (But not grandparents. They remain) 
    return reduced_potential 

def align_dimensions(parent_p, child_p, variable_topological_order):
    '''Reorder the child dimensions consistent with the parent, and return the
    transpose to apply to the child CPT. Since node removal 
    Before joining the two potentials need to add (unsqueeze) new dims 
    where they don't match.'''
    # promote_conditional_dimension()
    promote_conditional_dimension( init_permutation = list(range(9)))
    # How are ranks among tensors aligned.  
    # Note that each Potential has only one marginal, at the end of the dim list.
    # Test that the conditionings and marginals dim list respect the topological order
    # Place size 1 (by unsqueezing) dimensions where there are mis-matches in the dim order
    # Does this avoid having to transpose dimensions? 
    # Any size 1 dimensions in both lists for the same var can be removed. 
    # What about ordering of the resulting 2 marginals? When do they need to be swapped? 
    return updated_child_permutation, conditioning_var 

def join_parent(the_parent_p: Potential, the_child_p:Potential):
    '''Create a 2 dim joint by moving the parent dim to penultimate dim in 
    the child, if necessary.  Assume the marginal is always last in both
    potentials.
    '''
    parent_dims = the_parent_p.get_named_dims()
    child_dims = the_child_p.get_named_dims()
    # Find the permutation that puts the parent marginal dim second to last. 
    updated_child_permutation, conditioning_var = promote_conditional_dimension(parent_dims, 
                                     child_dims)
    if updated_child_permutation is None:
        # Its  not a parent of child, do nothing
        return None
    # TODO - this just tests if the permutation reorders the child. 
    elif (updated_child_permutation != list(range(the_child_p.rank()))):
        # If its not an identity permutation, apply it.
        the_child_p = the_child_p.permute_named_tensor(updated_child_permutation) 
    # TODO Where is the check that the parent marginal aligns with the child conditionings?
    # Find any common conditionings and move them before the two marginals
    # Place any child conditionings before the common conditionings
    # Place any parent conditionings before all
    # Compute the join. 
    return

def promote_conditional_dimension(parent_dims, child_dims, destined_position = 1):
    '''If the matching last dimension of the parent - assumed
    to be the marginal is not at the end of the list of conditionings
    for the child, find the transpose to move it.
    destined position:=
        1 to place the parent var to the last position in the child
        2 to place it second to last
    Return - the permutation to update child named_dims, putting the conditioning var at the end 
             of the list of conditionings,
           - the name of the conditioning variable to be removed.'''
    # We assume the parent marginal is the last variable in the dimensions
    child_conditionals = [m_var[0] for m_var in child_dims.items() if m_var[1] == 'c']
    parent_marginal = [c_var[0] for c_var in parent_dims.items() if c_var[1] in ('m', 'v')][0]  #TODO 'v' can never be a parent. 
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
            i_child_conditionals = list(child_dims.keys()).index(parent_marginal)
            # move it to the end of the child named_dims
            cc_permutation = swap_indexes(i_child_conditionals, 
                                          # where place the matching conditional
                                          len(cc_permutation)-destined_position, 
                                          cc_permutation) 
            # update the conditioning var
            conditioning_var = parent_marginal              
        return cc_permutation, conditioning_var
    
def swap_indexes(i_1, i_2, indexes):
    indexes[i_1], indexes[i_2] = indexes[i_2], indexes[i_1]
    return indexes 

def condition_decision(the_decn:Potential, the_observation: Potential) -> Potential:
    '''Add a dimension to increase the rank of the decision 
     to accommodate the conditioning variable.  The added dimension
     replicates the tensor along a  new dimension, using the marginal
     name of the conditioning variable as a new decision conditioning variable.'''
    # TODO What if there are multiple conditioning vars?
    # To modify the model inflight for VOI - computations, versus creating a new model.
    decn_dims = the_decn.get_named_dims().copy()
    decn_cpt = the_decn.cpt
    # Get new dimension name
    conditioning_var = the_observation.get_marginal_name()
    # Create new named dimension
    new_dims = OrderedDict([(conditioning_var, 'c')])
    new_dims.update(decn_dims)
    # Conditioning size
    last_dim_size = the_observation.get_dim_sizes()[-1]
    # Extend the decn tensor
    new_cpt = torch.stack(last_dim_size * [decn_cpt])
    return Potential(new_cpt,new_dims)

def condition_probability(joint_p: Potential, conditioning_p: Potential) -> Potential:
    'Normalize the joint by dividing by the conditiong var. Used for arc reversal'
    conditioned_p = joint_p / conditioning_p.unsqueeze(-1)
    conditioned_dims = joint_p.get_named_dims()
    # TODO preserve other conditioning vars for both Potentials
    return Potential(conditioned_p, conditioned_dims)

    
def marginalize_utility(u_potential: Potential, its_parent:Potential) -> Potential:
    '''Apply a potential assumed to be without parents that conditions the utility 
    to obtain the expected utility. The utility may have other parents, so the result is conditional
    on the remaining parents. '''
    unpeeled_utility = drop_final_singleton_dimension(u_potential)
  
    # Check that the parent is well formed - m,v,or d in last place only. 
    if not its_parent.check_last_dim():
        print('Error: parent {its_parent.get_marginal_name() is not well formed.}')
    parent_marginal = its_parent.get_marginal_name()
    # Is the parent marginal last? 
    if unpeeled_utility.get_dim_names()[-1] != parent_marginal:
        # Permute the result to promote the parent to the last place
        # in u_potential
        child_permutation, conditioning_var =\
              promote_conditional_dimension(its_parent.get_named_dims(), unpeeled_utility.get_named_dims())
        # assert conditioning_var == parent_marginal
        unpeeled_utility = unpeeled_utility.permute_named_tensor(child_permutation) 
    # join cpts
    joint = unpeeled_utility.cpt * its_parent.cpt
    cond_expected_utility = joint.sum(-1)
    reduced_dims = unpeeled_utility.remove_dim(parent_marginal)
    return Potential(cond_expected_utility, reduced_dims)

def maximize_utility(expected_utility: Potential, decision_p: Potential) -> tuple[Potential, Potential]:
    '''Find the maximum utility over a decision variable.
    This operation is used to optimize a decision based on expected utility.
    It removes the decision variable from the potential, retaining the maximum
    utility value for each combination of other conditioning variables.

    Returns a tuple of two Potentials:
    1. max_utility_potential: The utility potential with the decision variable marginalized out by maximization.
    2. policy_potential: A potential representing the optimal policy, i.e., the index of the
       decision state that yields the maximum utility for each conditioning state.
       (This should replace the decision variable potential. )
    '''
    # TODO run this without a conditiong var. 
    # Check that the dims of the decision_p match the utility 
    unpeeled_utility = drop_final_singleton_dimension(u_potential) 
    if unpeeled_utility.get_dim_sizes() != decision_p.get_dim_sizes():
        print(f'Error: utility shape {unpeeled_utility.get_dim_sizes()} does not match {decision_p.get_dim_sizes()}')
        return None, None
    # Find the last decision variable as the one to maximize over. 
    decision_var = decision_p.get_marginal_name()
    # TODO Check if the utility dims need to move the decn var to last place 
    decision_dim_idx = unpeeled_utility.find_var(decision_var)
    max_utility_values, policy_indicies = torch.max(expected_utility.cpt, dim=decision_dim_idx)

    max_utility_dims = unpeeled_utility.remove_dim(decision_var)
    max_utility_potential = Potential(max_utility_values, max_utility_dims)
    # TODO Reflate the policy indicies as a 0 - 1 matrix. Use the decn named dims instead
    max_policy = torch.zeros(decision_p.get_dim_sizes())
    for choice in range(list(decision_p.cpt.shape)[0]):  #TODO Assumes only one conditioning var. 
        max_policy[choice, policy_indicies[choice]] = 1

    decision_p.policy = max_policy 
    return max_utility_potential, decision_p
    

def join(parent_p, child_p): # parent, child
    '''Lower level function to multiply potentials once dimensions
    are aligned. Creates a joint of the parent and child.'''
    # Add a dimension at the end of the parent, so that it's marginal 
    # aligns with the same var that is the last condiioning
    # var of the child
    cpt = (parent_p.cpt.unsqueeze(-1) * child_p.cpt)
    # TODO extend to a parent with conditionings. 
    # For now assume only the child has conditionings
    child_named_dim = child_p.get_named_dims().copy()
    # The conditioning var is now also a marginal, assume its second to last
    child_named_dim[parent_p.get_dim_names()[-1]] = 'm'
    return Potential(cpt, child_named_dim)
    
def marginalize_out(the_potential, absorbed_var):
    '''Lower level function to marginalize a potential by 
    removing a dimension.  Typically used after a join. Marginalize
    out the parent to remove it, or the child, to reverse the arc.'''
    # remove the conditioning variable 
    # -- assumed to be the last (marginal) var of the parent.
    absorbed_dim = the_potential.find_var(absorbed_var)
    new_potential = the_potential.cpt.sum(absorbed_dim)
    named_sh = the_potential.remove_dim(absorbed_var)
    return Potential(new_potential, named_sh) 

### Utility Functions ###
# No problem with mapping single arg functions over tensors! 
def named_tensor_apply(a_potential, transformation, **kwargs): 
    dims = a_potential.get_named_dims()
    # Apply the utility function
    a_tensor = a_potential.cpt 
    u = transformation(a_potential.cpt, **kwargs)
    # u = 4/3*(1 - pow(kwargs['exponand'], (a_potential.cpt/kwargs['normalize'])))
    return Potential(u, dims)

def drop_final_singleton_dimension(the_potential):
    'Used to reduce utility for taking expectations'
    # As an alternative, don't add the singleton in the first place
    # Find the singleton dim, assuming only one singleton index,
    # not necessarily the last dimension. 
    # TODO - I suppose the only reason to have the singleton dimension
    #        is to mark tensors that are values, not probs. 
    final_dim = the_potential.get_dim_sizes()[-1]
    if final_dim != 1:
        # Presume the utility has already been pruned of the final dim 
        return the_potential
    else:
        # Get the name of that singleton dim
        singleton_name = the_potential.get_dim_names()[-1]
        # Remove it from the dims
        reduced_dims = the_potential.get_named_dims().copy()
        del reduced_dims[singleton_name]
    # x = the_potential.cpt.squeeze(1)
    return Potential(the_potential.cpt.squeeze(-1), reduced_dims)

# def delta_utility(a_value, **kwargs):
#     'Transformation from value to utility'
#     # dims = a_potential.get_named_dims()
#     a_utility = 4/3*(1 - pow(kwargs['exponand'], (a_value/kwargs['normalize'])))
#     return a_utility 

# def delta_inverse_utility(a_utility, **kwargs):
#     'Inverse transformation from utility back to value'
#     a_value = kwargs['normalize'] * np.log(1 - 3*a_utility/4)/np.log(kwargs['exponand'])
#     return a_value

### Main

if __name__ == '__main__':

    n_options = 3
    decn = new_Potential(n_options *[1/n_options], [n_options], ['location'])
    decn.pr_potential()
    print()

    probs = []
    for p in [0.9, 0.3]:
        for r in (p, 1 - p - 0.1, 0.1):
            probs.append(round(r,3))

    # probs = [ r for p in [0.9,  0.1, 0.0,  1.0, 0.3, 0.7] for r in (p, 1-p)]
    # Place margin probabilities in the last dimension
    child = new_Potential(probs, 
                       [2,3], 
                       ['condition1', 'uncertainty'])
    print('child:')
    child.pr_potential()
    print()

    parent = new_Potential([0.2, 0.8],
                            [2],
                            ['predictor'])
    print('\nParent: ')
    parent.pr_potential()

    new_decn = condition_decision(decn, parent)
    print('\nConditioned decision')
    new_decn.pr_potential()

    utils = [1, 10,9, 6,6,0]
    u_potential = new_Potential(utils, [2, n_options,  1], ['uncertainty', 'location', 'value'])
    u_potential.pr_potential()
    print()

    ### decision 
    ex_p, policy_p = maximize_utility(u_potential, new_decn)

    z = named_tensor_apply(u_potential, delta_utility, exponand = 0.5, normalize = 50)
    print(z)
    drop_final_singleton_dimension(u_potential).pr_potential()
    print('\n')

    root_parent = new_Potential([0.7, 0.2, 0.1], [3], ['uncertainty'])
    print('parent')
    root_parent.pr_potential()
    print('expected utility: ')
    marginalize_utility(u_potential, root_parent).pr_potential()


 

    # TODO marginalize utility for conditioned parents. 
    # marginalize_utility(u_potential, child).pr_potential()




    two_var_conditioned_decn = condition_decision(new_decn, parent)
    print('\nConditioned on "parent" also: ')
    two_var_conditioned_decn.pr_potential()
    print()

    print('\nMarginalize parent from child')
    new_child = absorb_parent(parent, child)
    if new_child is not None:
        new_child.pr_potential()
