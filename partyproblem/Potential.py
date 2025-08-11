# Potential
# A class that extends pytorch tensors with named dimensions
#
#   JMA April 2025


# Create a labelled dimension object.
# An OrderedDict preserves the insertion order of entries. 
from collections import OrderedDict
import torch

DEBUG = 1

class Potential (object):  # An named tensor
    '''
    CPTs are dimensioned with their marginal probability in the last dimension
    and conditioning probabilities dimensions before them. The last dimension
    varies by column.  In general, for joint probabilities, the variables 
    making up the joint follow the conditioning variables in the shape list. 
    TODO: Should there be a flag to distinguish marginal variables?
          In IDs, a CPT always has one, and just one marginal dimension, which is the last in the tensor.
          Similarly a value node has a last dimension of size 1, and all others are conditionings. 
    TODO: Should marginal variables also have their state labels included?
    TODO: The variable name is kept at the node level. 
    '''

    def __init__(self, cpt, n_shape):
        ' cpt  - multidim tensor, named_shape: OrderedDict '
        self.p = cpt
        self.named_shape = n_shape
        self.dim_names = n_shape.keys()

    def __repr__(self):
        return str(self.named_shape) + '\n\t' + repr(self.p)
    
    def get_named_shape(self):
        'The ordered dict of name: dimension, ...'
        return self.named_shape
    
    def get_dim_names(self):
        'The names from the named_shape.'
        return self.named_shape.keys()
    
    def get_dim_sizes(self):
        'The size of each dimension in the potential tensor'
        # Note: no need to duplicate this info in the named shape. 
        return self.p.shape
    
    def get_conditionings(self):
        'the type of the variable'
        return self.named_shape.items()
    
    def pr_potential(self):
        # the_potential = self.get_potential(a_node)
        print(f'\tnamed tensor: {list(self.get_named_shape().items())}, {list(self.p.shape)}')
        print('      ', 
              str(self.p).replace('tensor(','').replace(')', ''))
    
def new_Potential(prob_list, shape_list, dim_names, conditionings = None):
    'factory for creating potential from parsed xml components'
    if conditionings is None:
        # The default conditionings set the last dimension as
        # the marginal variable, and all others as conditioning variables.
        conditionings = ((len(shape_list)-1) * ['c']) + ['m']
    p = torch.tensor(prob_list).reshape(shape_list)
    nsh = OrderedDict(zip(dim_names, conditionings))
    return Potential(p, nsh)

### ID node #############################################################
# TODO or inherit from (dict)
class ID_node (object):

    def __init__(self, the_name):
        ''
        self.label = the_name    # TODO Duplicated as potential marginal? 
        self.kind = ''
        self.parents = None
        # state size is used to create its potential, when reaping the model file.
        # It applies to the potential marginal.
        # But is not stored with the potential.  
        self.states = []
        self.potential = new_Potential([],[0], [] )   #  of type Potential
        self.positions = None   # x,y node centers.  Plotting info.

    ### Accessors

    def get_node_name(self):
        return self.label
    
    def get_kind(self) -> str:
        return self.kind

    def get_parents(self):
        return self.parents
    
    def get_states(self):
        return self.states

    def state_size(self):
        return len(self.states) 
    
    def get_potential(self) -> Potential:
        return self.potential
    
    ### setters 
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

    def check_node(self):
        'Consistency check on node contents'
        if DEBUG: 
            print('Not implemented')

    ### Print
    def pr_node(self):
        print(f'{self.label}: {self.kind}')
        print(f'\tstates: {self.get_states()}')
        if self.potential is not None:
            self.potential.pr_potential()


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
    md = new_Potential([0.9,  0.1, 0.0,  1.0, 0.5, 0.5, 0.3, 0.7], 
                       [2,2,2], 
                       ['condition2', 'condition1', 'margin'])
    print('\n')
    md.pr_potential()
    print()

    features = dict(name='node1', 
                    kind='cpt', 
                    parents = [], 
                    states = ['False', 'True'],
                    potential = md)
    nd = create_from_dict(features)
    nd.pr_node()

#