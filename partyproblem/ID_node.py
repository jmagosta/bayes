#ID_node.py
#
#  JMA 14 Aug 2025

from Potential import *

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

    # Iterate twice thru the list. 
    probs = [ r for p in [0.9,  0.1, 0.0,  1.0, 0.3, 0.7] for r in (p, 1-p)]
    # Place margin probabilities in the last dimension
    md = new_Potential(probs,
                       [2,3,2], 
                       ['condition2', 'condition1', 'margin'])
    md.pr_potential()
    print()

    features = dict(name='node1', 
                    kind='cpt', 
                    parents = [], 
                    states = ['False', 'True'],
                    potential = md)
    nc = create_from_dict(features)
    nc.pr_node()

    # Test utilities
    d = new_Potential([1, 1], [2], 'decn1')
    features = dict(name='util1',
                    kind = 'decision',
                    parents = [],
                    states = ['No', 'Yes'],
                    potential = d)
    nd = create_from_dict(features)
    
    v = new_Potential([0.0, 1.1], [2,1], ['decn1', 'util'])
    features = dict(name='pref1', 
                    kind='utility', 
                    parents = [d], 
                    states = ['Value'],
                    potential = v)
    nu = create_from_dict(features)
    nu.pr_node()

    # Apply utility function
    from ID_operations import *   #TODO move delta_utility to potential? 
    u =  delta_utility(v)
    nu.potential = u

    nu.pr_node()


