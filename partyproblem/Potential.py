# Potential
# A class that extends pytorch tensors with named dimensions
#
#   JMA April 2025


# Create a labelled dimension object.
# An OrderedDict preserves the insertion order of entries. 
from collections import OrderedDict
import torch

class Potential (object):  # An named tensor
    '''
    CPTs are dimensioned with their marginal probability in the last dimension
    and conditioning probabilities dimensions before them. The last dimension
    varies by column.  In general, for joint probabilities, the variables 
    making up the joint follow the conditioning variables in the shape list. 
    TODO: Should there be a flag to distinguish marginal variables?
    TODO: Should marginal variables also have their state labels included?
    TODO: The variable name is kept at the node level. 
    '''

    def __init__(self, cpt, shape):
        ' cpt  - multidim tensor, shape: OrderedDict '
        self.p = cpt
        self.shape = shape
        self.dim_names = shape.keys()

    def __repr__(self):
        return str(self.shape) + '\n\t' + repr(self.p)
    
    def get_shape(self):
        'The ordered dict of name: dimension, ...'
        return self.shape
    
    def get_dim_names(self):
        'The names from the shape.'
        return self.shape.keys()
    
def new_Potential(prob_list, dim_list, dim_names ):
    'factory for creating potential from parsed xml components'
    p = torch.tensor(prob_list).reshape(dim_list)
    sh = OrderedDict(zip(dim_names, dim_list))
    return Potential(p, sh)

### ID node #############################################################
# TODO or inherit from (dict)
class ID_node (object):

    def __init__(self, the_name):
        ''
        self.label = the_name
        self.kind = ''
        self.parents = None
        # state size is used to create its potential, when reaping the model file.
        # It applies to the potential marginal.
        # But is not stored with the potential.  
        self.states = []
        self.potential = None
        self.positions = None   # x,y node centers.  Plotting info.

    def state_size(self):
        return len(self.states) 
    
    def get_parents(self):
        return self.parents
    
    def get_kind(self) -> str:
        return self.kind

    def pr_potential(self):
        # the_potential = self.get_potential(a_node)
        print('\tpotential: ', [k for k in self.potential.shape.keys()])
        print('      ', str(self.potential.p).replace('tensor(','').replace(')', ''))

    def pr_node(self):
        print(f'{self.label}: {self.kind}')
        if self.potential is not None:
            self.pr_potential()
    
    ## setters 
    def set_kind(self, the_kind):
        self.kind = the_kind

    def set_parents(self, the_parents):
        self.parents = the_parents

    def set_states(self, the_states):
        self.states = the_states

    def set_potential(self, the_potential):
        self.potential = the_potential

    def set_positions(self, the_positions):
        self.positions = the_positions

### Factory

def create_from_dict(the_features):
    ''
    the_node = ID_node(the_features['name'])
    the_node.set_kind (the_features['kind'])
    the_node.set_parents (the_features['parents'])
    the_node.set_states(the_features['states'])
    the_node.set_potential(the_features['potential'])
    return the_node

### Main

if __name__ == '__main__':

    # Place margin probabilities in the last dimension
    # md = new_Potential([1.0, 0.0, 0.1, 0.9, 0.0, 1.0, 0.4, 0.6], 
    #                    [2,2,2], 
    #                    ['condition1', 'condition', 'margin'])
    md = new_Potential([1.0,  0.1, 0.0,  0.4], 
                       [2,2,1], 
                       ['condition1', 'condition', 'margin'])
    print(md)

    nd = ID_node('node1')
    nd.potential = md
    print()
    nd.pr_node()

#