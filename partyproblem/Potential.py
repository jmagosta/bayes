# Potential
# A class that extends pytorch tensors with named dimensions
#
#   JMA April 2025


# Create a labelled dimension object.
# An OrderedDict preserves the insertion order of entries. 
from collections import OrderedDict
import torch

DEBUG = 1

class Potential (object):  # A named tensor
    '''
    CPTs are dimensioned with their marginal probability in the last dimension
    and conditioning probabilities dimensions before them. The last dimension
    varies by column.  In general, for joint probabilities, the variables 
    making up the joint follow the conditioning variables in the shape list. 
    Marginal variables are marked as 'm'
          In IDs, a CPT always has one, and just one marginal dimension, which is the last in the tensor.
          Similarly a value node has a last dimension of size 1, and all others are conditionings.

    Note that potentials do not know their parent nodes (just their conditionings) or their states.
    That info belongs with the ID_Node object. 

    TODO: Should marginal variables also have their state labels included?
    TODO: The variable name is kept at the node level, that can differ from the random variable name. 
    '''

    #type hints
    cpt: torch.Tensor
    named_dims: OrderedDict
    policy: torch.Tensor

    def __init__(self, cpt, n_dims):
        ' cpt  - multidim tensor, named_dims: OrderedDict '
        self.cpt = cpt
        self.named_dims = n_dims
        self.policy = None         #created by maximize utility. Same dims as the cpt

    def copy(self):
        'Use to avoid destructive operations'
        new_cpt = self.cpt.clone()
        new_dims = self.named_dims.copy()
        return Potential(new_cpt, new_dims)

    def __repr__(self):
        return str(self.named_dims) + '\n\t' + repr(self.cpt)
    
    def get_named_dims(self):
        'The ordered dict of name: dimension, ...'
        return self.named_dims
    
    def rank(self):
        'How many variables are there?'
        #TODO Should agree with the length of the tensor shape
        return len(self.named_dims)
    
    def get_dim_names(self):
        'The list of names from the named_dims.'
        return list(self.named_dims.keys())
    
    def get_dim_sizes(self):
        'The size of each dimension in the potential tensor'
        # Note: no need to duplicate this info in the named shape. 
        return self.cpt.shape
    
    def get_items(self):
        'the named dimensions as a list, to get the type of the variable'
        return self.named_dims.items()
    
    def get_var_conditioning(self, the_var: str):
        'get the label, c, v, d, or m from the variable string name. '
        return self.named_dims[the_var]
    
    def get_var_by_type(self, the_type = ('m')):
        'A list of vars selected by type'
        return [v for v,p in self.get_named_dims().items() if p in the_type]
    
    def get_marginal_name(self):
        return self.get_var_by_type(the_type= ('m', 'v', 'd'))[0]
    
    def find_var(self, the_var:str):
        'Look up the index by variable name'
        return list(self.named_dims.keys()).index(the_var)
    
    def index_named_dims(self, index_: int):
        'Return the dimension item by index'
        return list(self.named_dims.items())[index_]
    
    def check_last_dim(self):
        '''Check the actual variable follows the list of conditioning variables,
        and there is only one actual, i. e. this is not a join distribution.
        Actuals can be m, v, or d. Equivalently this is a test ID nodes.'''
        outcome = True
        var_types = [tp[1] for tp in list(self.get_items())] 
        if var_types[-1] not in ('m', 'v', 'd'):
            outcome = False
        if set(var_types[:-1]).intersection(set(('m', 'v', 'd'))):
            outcome = False
        return outcome

    def permute_named_dims(self, permutation):
        nd = list(self.named_dims.copy().items())
        return OrderedDict([nd[k] for k in permutation])
    
    def remove_dim(self, the_var:str):
        'Non destructive deletion'
        cp = self.named_dims.copy()
        del cp[the_var]
        return cp
    
    # Destructive change of the Potential
    def permute_named_tensor(self, permutation):
        'Reorder both tensor and dimensions by the permutation'
        self.named_dims = self.permute_named_dims(permutation)
        self.dim_names = self.named_dims.keys()
        self.cpt = torch.permute(self.cpt, permutation)
        return self
    
    def add_unit_dim(self, position, label):
        'Modify the named dimensions with an addition unit dimension'
        # Necessary to multiply together two Potentials
        self.cpt = self.cpt.unsqueeze(position)
        
###  print ###
    def pr_potential(self):
        # the_potential = self.get_potential(a_node)
        print(f'\tnamed tensor: {list(self.get_named_dims().items())}, {list(self.cpt.shape)}')
        print('      ', 
              str(self.cpt).replace('tensor(','').replace(')', ''))
    
### Factory ###
def new_Potential(prob_list, shape_list, dim_names, conditionings = None, the_var = 'm'):
    'factory for creating potential from parsed xml components'
    if conditionings is None:
        # The default conditionings sets the last dimension as
        # the marginal variable, and all others as conditioning variables.
        # TODO use "d" and "v" as conditioning types
        conditionings = ((len(shape_list)-1) * ['c']) + [the_var]
    p = torch.tensor(prob_list).reshape(shape_list)
    nsh = OrderedDict(zip(dim_names, conditionings))
    return Potential(p, nsh)

# No problem with mapping single arg functions over tensors!  
# # TODO cf potential operations delta_utility
# def delta_utility(x, exponand = 0.5, normalize = 50):
#     dims = x.get_named_dims()
#     u = 4/3*(1 - pow(exponand, (x.cpt/normalize)))
#     return Potential(u, dims)

def delta_utility(a_value, **kwargs):
    'Transformation from value to utility'
    # dims = a_potential.get_named_dims()
    a_utility = 4/3*(1 - pow(kwargs['exponand'], (a_value/kwargs['normalize'])))
    return a_utility 

def delta_inverse_utility(a_utility, **kwargs):
    'Inverse transformation from utility back to value'
    a_value = kwargs['normalize'] * np.log(1 - 3*a_utility/4)/np.log(kwargs['exponand'])
    return a

### Main #######################################################
if __name__ == '__main__':

    m1 = Potential(torch.tensor([0.2, 0.8]), OrderedDict([('a_var','m')]))
    m1.pr_potential()

# Iterate twice thru the list. 
    probs = [ r for p in [0.9,  0.1, 0.0,  1.0, 0.3, 0.7] for r in (p, 1-p)]
    # Place margin probabilities in the last dimension
    md = new_Potential(probs, 
                       [2,3,2], 
                       ['condition2', 'condition1', 'margin'])
    
    print('Check last dim:', md.check_last_dim())

    print('\n')
    md.pr_potential()
    print('marginal: ',md.get_marginal_name())
    print(f'rank: {md.rank()}')
    print()

    md_copy = md.copy()
    print(md_copy.permute_named_tensor((2,0,1)))

    # Test named_dim ops.
    print(md.get_var_conditioning('margin'))
    print(md.find_var('margin'))
    print(md.index_named_dims(1))
    print(md.permute_named_dims([2,1,0]))
    print(md.remove_dim('margin'))



#