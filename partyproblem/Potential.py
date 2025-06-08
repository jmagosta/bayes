# Potential
# A class that extends pytorch tensors with named dimensions
#
#   JMA April 2025


# Create a labelled dimension object.
from collections import OrderedDict
import torch

class Potential:

    def __init__(self, cpt, shape):
        ' cpt  - multidim tensor, shape: OrderedDict '
        self.p = cpt
        self.shape = shape
        self.dim_names = shape.keys()

    def __repr__(self):
        return str(self.shape) + '\n\t' + repr(self.p)
    
def new_Potential(prob_list, dim_list, dim_names ):
    'factory for creating potential from parsed xml components'
    p = torch.tensor(prob_list).reshape(dim_list)
    sh = OrderedDict(zip(dim_names, dim_list))
    return Potential(p, sh)
    
def get_potential(a_node, n_dict):
    'Find the probability np array in the node, and label it using parents in the graph'
    # The states of the RV label the columns, so that the matrix is row-markov
    the_cpt = n_dict[a_node]['potential']
    return the_cpt

### Main

if __name__ == '__main__':

    # Place margin probabilities in the last dimension
    md = new_Potential([0.1, 0.9, 0.4, 0.6], [2,2], ['condition', 'margin'])
    print(md)

#