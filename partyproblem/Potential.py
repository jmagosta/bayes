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
    TODO: Should marginal variables also have their state labels included?
    TODO: The variable name is kept at the node level, that can differ from the random variable name. 
    '''

    def __init__(self, cpt, n_shape):
        ' cpt  - multidim tensor, named_dims: OrderedDict '
        self.cpt = cpt
        self.named_dims = n_shape
        self.dim_names = n_shape.keys()  # remove?  TODO

    def __repr__(self):
        return str(self.named_dims) + '\n\t' + repr(self.cpt)
    
    def get_named_dims(self):
        'The ordered dict of name: dimension, ...'
        return self.named_dims
    
    def get_dim_names(self):
        'The names from the named_dims.'
        return self.named_dims.keys()
    
    def get_dim_sizes(self):
        'The size of each dimension in the potential tensor'
        # Note: no need to duplicate this info in the named shape. 
        return self.cpt.shape
    
    def get_items(self):
        'the named dimensions as a list, to get the type of the variable'
        return self.named_dims.items()
    
    def get_var_conditioning(self, the_var: str):
        'get the label, c or m from the variable string name. '
        return self.named_dims[the_var]
    
    def find_var(self, the_var:str):
        'Look up the index by variable name'
        return list(self.named_dims.keys()).index(the_var)
    
    def index_named_dims(self, index_: int):
        'Return the dimension item by index'
        return list(self.named_dims.items())[index_]
    
    def permute_named_dims(self, permutation):
        nd = list(self.named_dims.copy().items())
        return OrderedDict([nd[k] for k in permutation])
    
    def remove_dim(self, the_var:str):
        'Non destructive deletion'
        cp = self.named_dims.copy()
        del cp[the_var]
        return cp
    
    # Destructive change of the Potential
    def transpose_named_tensor(self, permutation):
        'Reorder both tensor and dimensions by the permutation'
        self.named_dims = self.permute_named_dims(permutation)
        self.dim_names = self.named_dims.keys()
        self.cpt = torch.permute(self.cpt, permutation)
        return self
        
###  print ###
    def pr_potential(self):
        # the_potential = self.get_potential(a_node)
        print(f'\tnamed tensor: {list(self.get_named_dims().items())}, {list(self.cpt.shape)}')
        print('      ', 
              str(self.cpt).replace('tensor(','').replace(')', ''))
    
### Factory ###
def new_Potential(prob_list, shape_list, dim_names, conditionings = None):
    'factory for creating potential from parsed xml components'
    if conditionings is None:
        # The default conditionings set the last dimension as
        # the marginal variable, and all others as conditioning variables.
        conditionings = ((len(shape_list)-1) * ['c']) + ['m']
    p = torch.tensor(prob_list).reshape(shape_list)
    nsh = OrderedDict(zip(dim_names, conditionings))
    return Potential(p, nsh)

### Main #######################################################
if __name__ == '__main__':

# Iterate twice thru the list. 
    probs = [ r for p in [0.9,  0.1, 0.0,  1.0, 0.3, 0.7] for r in (p, 1-p)]
    # Place margin probabilities in the last dimension
    md = new_Potential(probs, 
                       [2,3,2], 
                       ['condition2', 'condition1', 'margin'])
    print('\n')
    md.pr_potential()
    print()

    print(md.transpose_named_tensor((2,0,1)))

    # Test named_dim ops.
    print(md.get_var_conditioning('margin'))
    print(md.find_var('margin'))
    print(md.index_named_dims(1))
    print(md.permute_named_dims([2,1,0]))
    print(md.remove_dim('margin'))



#